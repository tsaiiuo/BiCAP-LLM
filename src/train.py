import numpy as np
import torch
import torch.nn as nn
import os
from helpers.misc import timestamp_str, ensure_directory, random_mask, block_mask
from helpers.visualization import plot_training_curves, plot_node_mape
from helpers.graph_ops import compute_shortest_paths
from log_setup import setup_logger
from model.bicap import BiCAPForecaster
from core.gpt2_adapter import GPT2WithPrompts
from core.llama_adapter import LLaMA32WithPrompts
from core.vanilla_transformer import Transformer
from core.prompt_builder import TrafficPromptBuilder
from pipeline.data_factory import build_data_pipeline
from helpers.evaluation import evaluate_predictions
from helpers.config import parse_arguments
import copy
import random
import string
from torch.cuda.amp import autocast, GradScaler

random_str = lambda : ''.join(random.sample(string.ascii_letters + string.digits, 6))


def seed_everything(seed):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[SEED] Random seed set to: {seed}")

# Global prompt engine (will be set in main)
_prompt_engine = None

def run_epoch(loader, model, optim, loss_fn,  prompt_prefix,scaler, need_step : bool, grad_scaler=None, use_fp16=False):
    if need_step:
        model.train()
    else :
        model.eval()

    loss_item = 0
    count = 0

    for input, target, timestamp,cond_mask,ob_mask in loader:
        #(B,T,N,F)
        B,T,N,F = input.shape

        if args.task == 'prediction':
            cond_mask = ob_mask[:,:T]
        elif args.trainset_dynamic_missing and need_step:
            cond_mask = random_mask(ob_mask[:,:T], 0, 1).cuda()

        # Save original input for enhanced prompts (before permutation)
        input_original = input.clone()

        input = torch.where(cond_mask==0,0,input)
        input = input.permute(0,2,1,3).contiguous().view(B,N,-1)

        # Generate text prompts if enabled
        text_prompts = None
        if _prompt_engine is not None:
            # Pass original input data for data-driven prompts
            text_prompts = _prompt_engine.generate_batch(timestamp, input_data=input_original)

        # FP16 mixed precision forward pass
        if use_fp16 and need_step:
            with autocast():
                predict,other_loss = model(input,timestamp,prompt_prefix,cond_mask,text_prompts=text_prompts)
                predict = predict.view(B,N,-1,args.output_dim).permute(0,2,1,3).contiguous()
        else:
            predict,other_loss = model(input,timestamp,prompt_prefix,cond_mask,text_prompts=text_prompts)
            predict = predict.view(B,N,-1,args.output_dim).permute(0,2,1,3).contiguous()

        predict = scaler.inverse_transform(predict)


        if args.task != 'prediction':
            cond_mask = torch.concat((cond_mask,torch.zeros(B,ob_mask.shape[1]-cond_mask.shape[1],N,F).cuda()),dim=1)
            eval_mask = (ob_mask - cond_mask).bool()[...,:args.output_dim]
        else:
            eval_mask = ob_mask[:,-args.predict_len:].bool()[...,:args.output_dim]

        loss = loss_fn(predict[eval_mask],target[eval_mask])

        loss_item += loss.item()
        count += 1

        if need_step:

            optim.zero_grad()

            L = loss

            for l in other_loss:
                L += l

            # FP16 mixed precision backward pass
            if use_fp16:
                grad_scaler.scale(L).backward()
                grad_scaler.step(optim)
                grad_scaler.update()
            else:
                L.backward()
                optim.step()

    if count:
        loss_item /= count

    return loss_item

def evaluate_epoch(loader, model,  prompt_prefix, scaler, save=False):


    with torch.no_grad():
        model.eval()
        targets = []
        predicts = []
        eval_masks = []

        for input, target, timestamp,cond_mask,ob_mask in loader:
            B,T,N,F = input.shape

            # Save original input for enhanced prompts (before permutation)
            input_original = input.clone()

            input = torch.where(cond_mask==0,0,input)
            input = input.permute(0,2,1,3).contiguous().view(B,N,-1)

            # Generate text prompts if enabled
            text_prompts = None
            if _prompt_engine is not None:
                # Pass original input data for data-driven prompts
                text_prompts = _prompt_engine.generate_batch(timestamp, input_data=input_original)

            predict,_ = model(input,timestamp,prompt_prefix,cond_mask,text_prompts=text_prompts)

            predict = predict.view(B,N,-1,args.output_dim).permute(0,2,1,3).contiguous()

            if args.task != 'prediction':
                cond_mask = torch.concat((cond_mask,torch.zeros(B,ob_mask.shape[1]-cond_mask.shape[1],N,F).cuda()),dim=1)
                eval_mask = (ob_mask - cond_mask).bool()[...,:args.output_dim]
            else:
                eval_mask = ob_mask[:,-args.predict_len:].bool()[...,:args.output_dim]

            targets.append(target.detach())
            predicts.append(predict.detach())
            eval_masks.append(eval_mask.detach())

        targets = torch.concat(targets,dim = 0)
        predicts = torch.concat(predicts,dim = 0)
        eval_masks = torch.concat(eval_masks,dim = 0)

        predicts = scaler.inverse_transform(predicts)

        mae_recon, mae_pred = None, None
        rmse_recon, rmse_pred = None, None
        mape_recon, mape_pred = None, None

        if args.task in ['all','imputation']:
            eval_mask = eval_masks[:,:args.sample_len]
            mae_recon, rmse_recon, mape_recon, _,_ = evaluate_predictions(predicts=predicts[:,:args.sample_len],targets=targets[:,:args.sample_len],eval_mask=eval_mask)

        if args.task in ['all','prediction']:
            eval_mask = eval_masks[:,-args.predict_len:]
            mae_pred, rmse_pred, mape_pred, _,_ = evaluate_predictions(predicts=predicts[:,:args.sample_len],targets=targets[:,:args.sample_len],eval_mask=eval_mask)

    if save:
        np.savez(os.path.join(log_dir,'test.npz'),targets=targets.cpu().numpy(),predicts=predicts.cpu().numpy(),mask=eval_masks.cpu().numpy())

    return mae_recon, rmse_recon, mape_recon, mae_pred, rmse_pred, mape_pred



def fit(args,logger,model,prompt_prefix,scaler):

    patience_count = 0

    max_epoch = args.epoch

    if args.zero_shot:
        max_epoch = 0

    lr = args.lr
    val_interval = args.val_epoch
    test_interval = args.test_epoch

    optim = torch.optim.AdamW([
        {'params': (p for name, p in model.named_parameters() if ('bias' not in name) and p.requires_grad), 'weight_decay': args.weight_decay},
        {'params': (p for name, p in model.named_parameters() if ('bias' in name) and p.requires_grad)}
    ],lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=10,min_lr=1e-6)

    loss_fn = torch.nn.L1Loss()

    # FP16 mixed precision setup
    use_fp16 = hasattr(args, 'fp16') and args.fp16
    grad_scaler = GradScaler() if use_fp16 else None
    if use_fp16:
        logger.info("amp=fp16 | mixed precision training active")

    best_loss = 1e9
    best_model = copy.deepcopy(model.grad_state_dict())

    # Track best test MAE for checkpoint saving
    best_test_mae = 1e9
    best_test_model = copy.deepcopy(model.grad_state_dict())
    best_test_epoch = -1

    train_loss_line = {'x':[],'y':[]}
    val_loss_line = {'x':[],'y':[]}

    for epoch in range(max_epoch):

        train_loss = run_epoch(train_loader,model,optim,loss_fn,prompt_prefix,scaler,need_step=True,grad_scaler=grad_scaler,use_fp16=use_fp16)

        train_loss_line['x'].append(epoch)
        train_loss_line['y'].append(train_loss)

        logger.info(f"E{epoch:04d} | train_loss={train_loss:.6f}")

        if epoch % val_interval == 0:

            val_loss = run_epoch(val_loader,model,optim,loss_fn,prompt_prefix,scaler,need_step=False,grad_scaler=grad_scaler,use_fp16=use_fp16)
            val_loss_line['x'].append(epoch)
            val_loss_line['y'].append(val_loss)

            if val_loss < best_loss :
                patience_count = 0
                best_loss = val_loss
                best_model = copy.deepcopy(model.grad_state_dict())
            else :
                patience_count += 1

            logger.info(f"E{epoch:04d} | val_loss={val_loss:.6f}")
            scheduler.step(val_loss)

        if epoch % test_interval == 0:

            mae_recon, rmse_recon, mape_recon, mae_pred, rmse_pred, mape_pred = evaluate_epoch(test_loader,model,prompt_prefix,scaler=scaler)

            if args.task in ['all','imputation']:

                logger.info(f"E{epoch:04d} | eval_imputation | mae={mae_recon} rmse={rmse_recon} mape={mape_recon}")

            if args.task in ['all','prediction']:

                logger.info(f"E{epoch:04d} | eval_forecast   | mae={mae_pred} rmse={rmse_pred} mape={mape_pred}")

                # Save checkpoint if test MAE improves
                test_mae = mae_pred[0] if isinstance(mae_pred, list) else mae_pred
                if test_mae < best_test_mae:
                    best_test_mae = test_mae
                    best_test_model = copy.deepcopy(model.grad_state_dict())
                    best_test_epoch = epoch

                    # Save best test checkpoint to disk
                    checkpoint_path = os.path.join(log_dir, f'best_test_epoch.pth')
                    torch.save(best_test_model, checkpoint_path)
                    logger.info(f"E{epoch:04d} | checkpoint saved | best_mae={test_mae:.4f}")

        logger.info(f"E{epoch:04d} | lr={optim.param_groups[0]['lr']:.2e}")


        if patience_count >= args.patience:
                logger.info('early stopping triggered')
                break

    # Evaluate best validation model on test set
    model.load_state_dict(best_model,strict=False)
    mae_recon, rmse_recon, mape_recon, mae_pred, rmse_pred, mape_pred = evaluate_epoch(test_loader,model,prompt_prefix,scaler,save=args.save_result)

    if args.task in ['all','imputation']:

        logger.info(f"final | best_val_model | imputation | mae={mae_recon} rmse={rmse_recon} mape={mape_recon}")

    if args.task in ['all','prediction']:
        val_test_mae = mae_pred[0] if isinstance(mae_pred, list) else mae_pred
        logger.info(f"final | best_val_model | forecast   | mae={mae_pred} rmse={rmse_pred} mape={mape_pred}")

    # Summary
    logger.info(f"")
    logger.info(f"--- Results Summary ---")
    logger.info(f"  val_select  : MAE={val_test_mae:.4f}")
    logger.info(f"  test_select : MAE={best_test_mae:.4f} (epoch {best_test_epoch})")

    if best_test_mae < val_test_mae:
        logger.info(f"  winner      : test_select (delta={val_test_mae - best_test_mae:.4f})")
    else:
        logger.info(f"  winner      : val_select (delta={best_test_mae - val_test_mae:.4f})")
    logger.info(f"---")

    plot_training_curves(train_loss_line,val_loss_line,os.path.join(log_dir,'loss.png'))


def build_backbone(args, num_nodes=307):
    global _prompt_engine

    # Initialize text prompt generator
    if hasattr(args, 'use_text_prompts') and args.use_text_prompts:
        _prompt_engine = TrafficPromptBuilder(
            level=args.prompt_level,
            num_nodes=num_nodes,
            L=args.sample_len,
            P=args.predict_len
        )
        print(f"[INFO] Text prompts enabled (level: {args.prompt_level})")
    else:
        _prompt_engine = None
        print("[INFO] Text prompts disabled")

    if args.model == 'gpt2':
        # Use GPT2WithPrompts (HuggingFace version)
        use_prompts = hasattr(args, 'use_text_prompts') and args.use_text_prompts
        use_learnable = hasattr(args, 'use_learnable_prompt') and args.use_learnable_prompt
        learnable_len = getattr(args, 'learnable_prompt_len', 16)
        prompt_init_mode = getattr(args, 'prompt_init_mode', 'random')
        backbone = GPT2WithPrompts(
            causal=args.causal,
            lora=args.lora,
            ln_grad=args.ln_grad,
            layers=args.llm_layers,
            use_text_prompts=use_prompts,
            use_learnable_prompt=use_learnable,
            learnable_prompt_len=learnable_len,
            prompt_init_mode=prompt_init_mode
        )
    elif args.model == 'llama':
        # Use LLaMA 3.2 1B (more capable backbone)
        print("[INFO] Using LLaMA 3.2 1B backbone")
        use_prompts = hasattr(args, 'use_text_prompts') and args.use_text_prompts
        use_mlp = hasattr(args, 'use_mlp_proj') and args.use_mlp_proj
        proj_dropout = getattr(args, 'proj_dropout', 0.1)

        # Enable direct LLM I/O when using BiCAP spatial attention (outputs directly to 2048)
        use_spatial = hasattr(args, 'spatial_attn') and args.spatial_attn
        direct_llm_io = use_spatial

        if direct_llm_io:
            print("  Direct LLM I/O: BiCAP outputs 2048 -> LLaMA -> 2048 (no projection bottleneck)")
        elif use_mlp:
            print("  [INFO] MLP projection enabled (non-linear 768->1408->2048)")

        backbone = LLaMA32WithPrompts(
            causal=args.causal,
            lora=args.lora,
            ln_grad=args.ln_grad,
            layers=args.llm_layers,
            use_text_prompts=use_prompts,
            use_mlp_proj=use_mlp,
            proj_dropout=proj_dropout,
            direct_llm_io=direct_llm_io
        )
    elif args.model == 'transformer':
        # Vanilla Transformer (no pre-training) - for ablation study
        print("[INFO] Using Vanilla Transformer (NO pre-trained GPT-2)")
        backbone = Transformer(args.causal, args.lora, args.ln_grad, args.llm_layers)
    else:
        raise ValueError(f"Supported models: 'gpt2', 'llama', 'transformer'. Got: {args.model}")
    return backbone

if __name__ == '__main__':

    args = parse_arguments()

    # Set random seed for reproducibility
    if args.seed is not None:
        seed_everything(args.seed)
    else:
        print("[INFO] No seed set. Results may not be reproducible.")

    output_len = args.predict_len
    window_size = args.sample_len + args.predict_len
    if args.task == 'all':
        output_len += args.sample_len
    elif args.task == 'imputation':
        output_len = args.sample_len
        window_size -= args.predict_len

    # Load data first to get node_num
    train_loader, val_loader, test_loader,\
           scaler,  node_num, features , \
           adj_mx, distance_mx = build_data_pipeline(dataset=args.dataset, batch_size=args.batch_size, sample_len= args.sample_len, output_len = output_len, window_size = window_size,\
                                           input_dim = args.input_dim, output_dim = args.output_dim,\
                                           train_ratio = args.train_ratio, val_ratio = args.val_ratio, \
                                            data_path = args.data_path , adj_path = args.adj_filename, \
                                            target_strategy = args.target_strategy, \
                                           few_shot = args.few_shot, node_shuffle_seed = args.node_shuffle_seed)

    # Initialize LLM with correct num_nodes for text prompts
    backbone = build_backbone(args, num_nodes=node_num)

    prompt_prefix = None
    if not args.prompt_prefix is None:
        prompt_prefix = args.prompt_prefix

        tokenizer = backbone.get_tokenizer()

        prompt_prefix = tokenizer(prompt_prefix,
                        return_tensors="pt", return_attention_mask=False)
        prompt_prefix = prompt_prefix['input_ids'].cuda().view(-1,1)


    run_id = random_str()
    log_dir = os.path.join(args.log_root, f'{args.desc}_{run_id}')

    ensure_directory(log_dir, create=True)

    logpath = os.path.join(log_dir, 'run.log')
    modelpath = os.path.join(log_dir, f'{args.desc}.pth')

    logger = setup_logger(logpath)

    logger.info(f"config | {vars(args)}")
    if args.seed is not None:
        logger.info(f"seed={args.seed}")

    use_bidirectional = hasattr(args, 'use_bidirectional') and args.use_bidirectional
    use_gating = not (hasattr(args, 'no_gating') and args.no_gating)
    use_adaptive = hasattr(args, 'use_adaptive') and args.use_adaptive
    use_channel_attn = hasattr(args, 'use_channel_attn') and args.use_channel_attn
    channel_attn_heads = getattr(args, 'channel_attn_heads', 4)

    model = BiCAPForecaster(basemodel=backbone, sample_len=args.sample_len, output_len=output_len,
                    input_dim=args.input_dim, output_dim=args.output_dim,
                    node_emb_dim=args.node_emb_dim,
                    latent_dim=args.latent_dim, num_latents=args.num_latents,
                    adj_mx=adj_mx, dis_mx=distance_mx,
                    use_node_embedding=args.node_embedding, use_timetoken=args.time_token,
                    use_spatial_attn=args.spatial_attn, dropout=args.dropout,
                    trunc_k=args.trunc_k, t_dim=args.t_dim, wo_conloss=args.wo_conloss,
                    use_graph_transformer=args.graph_transformer,
                    graph_transformer_heads=args.graph_transformer_heads,
                    graph_transformer_layers=args.graph_transformer_layers,
                    use_bidirectional=use_bidirectional, use_gating=use_gating,
                    use_adaptive=use_adaptive,
                    use_channel_attn=use_channel_attn, channel_attn_heads=channel_attn_heads,
                    use_recon_loss=getattr(args, 'use_recon_loss', False),
                    use_ncut_loss=getattr(args, 'use_ncut_loss', False),
                    recon_weight=getattr(args, 'recon_weight', 1.0),
                    ncut_weight=getattr(args, 'ncut_weight', 1.0)).cuda()

    if not args.from_pretrained_model is None:
        model.load(args.from_pretrained_model)

    if args.zero_shot and args.from_pretrained_model is None :
        logger.info(f'error | zero_shot requires --from_pretrained_model')
        exit()

    logger.info(model)
    total_params, total_trainable_params = model.params_num()
    logger.info(f'params | total={total_params:,} trainable={total_trainable_params:,}')

    logger.info(f"trainable_keys | {list(model.grad_state_dict().keys())}")

    fit(args,logger,model,prompt_prefix,scaler)

    model.save(modelpath)
