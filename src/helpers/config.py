"""Command-line argument parsing for BiCAP-GPT training."""

import argparse
from typing import Optional


def _model_options(parser: argparse.ArgumentParser) -> None:
    """Add model architecture arguments."""
    grp = parser.add_argument_group("Model Architecture")

    grp.add_argument("--lora", action="store_true",
                     help="Enable LoRA fine-tuning on attention layers")
    grp.add_argument("--ln_grad", action="store_true",
                     help="Make LayerNorm parameters trainable")

    # Semantic prompt options
    grp.add_argument("--use_text_prompts", action="store_true",
                     help="Enable text prompts for semantic understanding")
    grp.add_argument("--prompt_level", type=str, default="simple",
                     choices=["simple", "enhanced", "task", "task_enhanced"],
                     help="Text prompt level: simple, enhanced, task, or task_enhanced")

    # Learnable prompt (ablation baseline)
    grp.add_argument("--use_learnable_prompt", action="store_true",
                     help="Use learnable prompt embeddings instead of text prompts (for ablation)")
    grp.add_argument("--learnable_prompt_len", type=int, default=16,
                     help="Number of learnable prompt tokens (default: 16)")
    grp.add_argument("--prompt_init_mode", type=str, default="random",
                     choices=["random", "vocab", "text_init", "mlp"],
                     help="Learnable prompt initialization: random, vocab (from GPT-2 vocab), "
                          "text_init (from template text), mlp (Prefix-Tuning style reparameterization)")

    grp.add_argument("--causal", default=0, type=int,
                     help="LLM causal attention")
    grp.add_argument("--prompt_prefix", default=None, type=str,
                     help="Hard-coded prompt prefix text")
    grp.add_argument("--node_embedding", action="store_true")
    grp.add_argument("--time_token", action="store_true")

    grp.add_argument("--model", default="gpt2", type=str,
                     choices=["gpt2", "llama", "transformer"],
                     help="LLM backbone: gpt2 (default), llama (LLaMA 3.2 1B), transformer (vanilla)")
    grp.add_argument("--llm_layers", default=None, type=int)

    # LLaMA projection
    grp.add_argument("--use_mlp_proj", action="store_true",
                     help="Use MLP projection instead of linear for LLaMA (addresses 768->2048 bottleneck)")
    grp.add_argument("--proj_dropout", default=0.1, type=float,
                     help="Dropout for MLP projection layers (default: 0.1)")

    grp.add_argument("--dropout", default=0, type=float)
    grp.add_argument("--trunc_k", default=16, type=int)
    grp.add_argument("--t_dim", default=64, type=int)
    grp.add_argument("--node_emb_dim", default=128, type=int)

    # BiCAP spatial perceiver
    spatial = parser.add_argument_group("Spatial Attention (BiCAP)")
    spatial.add_argument("--spatial_attn", action="store_true",
                         help="Use BiCAP spatial attention (disabled if --graph_transformer is set)")
    spatial.add_argument("--wo_conloss", action="store_true")
    spatial.add_argument("--use_recon_loss", action="store_true",
                         help="Reconstruction loss for spatial compression (encode->decode fidelity)")
    spatial.add_argument("--use_ncut_loss", action="store_true",
                         help="Normalized cut loss for graph-aware spatial partitioning")
    spatial.add_argument("--recon_weight", default=1.0, type=float,
                         help="Weight for reconstruction loss (default: 1.0)")
    spatial.add_argument("--ncut_weight", default=1.0, type=float,
                         help="Weight for normalized cut loss (default: 1.0)")
    spatial.add_argument("--latent_dim", default=128, type=int,
                         help="BiCAP latent dimension for Q/K projection (default: 128)")
    spatial.add_argument("--num_latents", default=128, type=int,
                         help="Number of BiCAP latent tokens (default: 128)")

    # BiXT bidirectional cross-attention
    spatial.add_argument("--use_bidirectional", action="store_true",
                         help="Enable BiXT bidirectional cross-attention in BiCAP (NeurIPS'24)")
    spatial.add_argument("--no_gating", action="store_true",
                         help="Disable gating mechanism in BiXT")

    # Adaptive tokens
    spatial.add_argument("--use_adaptive", action="store_true",
                         help="Enable adaptive latent tokens in BiCAP (STAEformer-inspired, CIKM'23)")

    # Channel-wise attention
    spatial.add_argument("--use_channel_attn", action="store_true",
                         help="Enable channel-wise attention in BiCAP (iTransformer-inspired, ICLR'24)")
    spatial.add_argument("--channel_attn_heads", default=4, type=int,
                         help="Number of attention heads for channel attention (default: 4)")

    # Graph Transformer alternative
    gt = parser.add_argument_group("Graph Transformer")
    gt.add_argument("--graph_transformer", action="store_true",
                    help="Use Graph Transformer instead of BiCAP")
    gt.add_argument("--graph_transformer_heads", default=6, type=int,
                    help="Number of attention heads in Graph Transformer (default: 6)")
    gt.add_argument("--graph_transformer_layers", default=1, type=int,
                    help="Number of GCN layers in Graph Transformer (default: 1)")


def _data_options(parser: argparse.ArgumentParser) -> None:
    """Add dataset and I/O arguments."""
    grp = parser.add_argument_group("Data")
    grp.add_argument("--dataset", type=str)
    grp.add_argument("--data_path", type=str)
    grp.add_argument("--adj_filename", default=None, type=str)
    grp.add_argument("--sample_len", default=12, type=int)
    grp.add_argument("--predict_len", default=12, type=int)
    grp.add_argument("--train_ratio", default=0.6, type=float)
    grp.add_argument("--val_ratio", default=0.6, type=float)
    grp.add_argument("--input_dim", default=1, type=int)
    grp.add_argument("--output_dim", default=1, type=int)


def _training_options(parser: argparse.ArgumentParser) -> None:
    """Add training hyper-parameter arguments."""
    grp = parser.add_argument_group("Training")
    grp.add_argument("--lr", default=0.001, type=float)
    grp.add_argument("--weight_decay", default=0.05, type=float)
    grp.add_argument("--batch_size", default=4, type=int)
    grp.add_argument("--epoch", default=100, type=int)
    grp.add_argument("--val_epoch", default=5, type=int)
    grp.add_argument("--test_epoch", default=5, type=int)
    grp.add_argument("--patience", default=100, type=int)
    grp.add_argument("--fp16", action="store_true",
                     help="Enable FP16 mixed precision training (30-50%% speedup)")


def parse_arguments() -> argparse.Namespace:
    """Parse all command-line arguments and return the namespace."""
    parser = argparse.ArgumentParser(description="BiCAP-GPT Traffic Forecaster")

    parser.add_argument("--desc", default="bicap_gpt", type=str, help="Experiment description")
    parser.add_argument("--log_root", default="../logs", type=str, help="Log root directory")
    parser.add_argument("--from_pretrained_model", default=None, type=str)
    parser.add_argument("--zero_shot", action="store_true")
    parser.add_argument("--save_result", action="store_true")
    parser.add_argument("--few_shot", default=1, type=float)
    parser.add_argument("--seed", default=None, type=int,
                        help="Random seed for reproducibility. If None, no seed is set.")
    parser.add_argument("--node_shuffle_seed", default=None, type=int)

    parser.add_argument("--task", default="prediction", type=str,
                        help="Task: prediction, imputation, or all")
    parser.add_argument("--target_strategy", default="random", type=str,
                        help="Masking strategy for imputation: random or hybrid")
    parser.add_argument("--trainset_dynamic_missing", action="store_true",
                        help="Enable dynamic missing pattern in training set")

    _data_options(parser)
    _model_options(parser)
    _training_options(parser)

    return parser.parse_args()
