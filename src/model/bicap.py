import torch
import torch.nn as nn
from typing import Optional
from helpers.graph_ops import normalize_adjacency, compute_laplacian_eigenvectors, compute_node_ordering
from core.bicap_module import BCSPerceiver
from core.graph_transformer import SpatialTemporalGraphTransformer, convert_adj_to_sparse
import numpy as np
from core.sinusoidal_pe import SinusoidalPositionEncoder


class PredictionHead(nn.Module):
    """MLP decoder: LLM hidden → prediction output."""

    def __init__(self, input_dim, emb_dim, output_dim):
        super().__init__()
        hidden_size = (emb_dim + output_dim) * 2 // 3
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, llm_hidden):
        return self.fc(llm_hidden)

class TemporalEncoder(nn.Module):
    """Encode timestamp into dense embedding.

    Input TE shape: (B, T, 5) = [month, day_of_month, day_of_week, hour, minute]
    Output shape: (B, T, 2 * t_dim)
    """

    def __init__(self, t_dim):
        super().__init__()
        self.day_embedding = nn.Embedding(num_embeddings=288, embedding_dim=t_dim)   # 5-min slots per day
        self.week_embedding = nn.Embedding(num_embeddings=7, embedding_dim=t_dim)    # day of week

    def forward(self, TE):
        B, T, _ = TE.shape

        week = (TE[..., 2].to(torch.long) % 7).view(B*T, -1)
        hour = (TE[..., 3].to(torch.long) % 24).view(B*T, -1)
        minute = (TE[..., 4].to(torch.long) % 60).view(B*T, -1)

        DE = self.day_embedding((hour * 60 + minute) // 5)
        WE = self.week_embedding(week)

        te = torch.concat((DE, WE), dim=-1).view(B, T, -1)
        return te


class SpectralNodeEncoder(nn.Module):
    def __init__(self, adj_mx, node_emb_dim, k = 16, dropout = 0 ):
        super().__init__()
        N,_ = adj_mx.shape
        self.k = k

        self.setadj(adj_mx=adj_mx)

        self.fc = nn.Linear(in_features=k,out_features=node_emb_dim)

    def forward(self):

        node_emgedding = self.fc(self.lap_eigvec)

        return node_emgedding
    
    def setadj(self,adj_mx):
        N,_ = adj_mx.shape

        self.adj_mx = adj_mx

        eigvec, eigval = compute_laplacian_eigenvectors(self.adj_mx)
        k = self.k
        if k>N:
            eigvec = np.concatenate((eigvec, np.zeros(N,k-N)), dim = -1)
            eigval = np.concatenate((eigval, np.zeros(k-N)), dim = -1)
        
        ind = np.abs(eigval).argsort(axis=0)[::-1][:k]

        eigvec = eigvec[:, ind]        

        if hasattr(self,'lap_eigvec'):
            self.lap_eigvec = torch.tensor(eigvec).float()
        else :
            self.register_buffer('lap_eigvec', torch.tensor(eigvec).float())
    
class TemporalTokenizer(nn.Module):
    """Create 2 global temporal tokens: state (current snapshot) + gradient (trend).

    Input: x (B, N, T*F), te (B, T, tim_dim), mask (B, T, N, F)
    Output: (B, 2, output_dim)
    """

    def __init__(self, sample_len, features, emb_dim, tim_dim, dropout, output_dim=None):
        super().__init__()

        self.sample_len = sample_len
        self.output_dim = output_dim if output_dim is not None else emb_dim

        in_features = sample_len * features * 2 + tim_dim
        hidden_size = (in_features + self.output_dim) * 2 // 3
        self.fc_state = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.output_dim),
        )

        grad_features = tim_dim + (sample_len - 1) * features * 2
        hidden_size = (grad_features + self.output_dim) * 2 // 3
        self.fc_grad = nn.Sequential(
            nn.Linear(grad_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.output_dim),
        )

        self.ln = nn.LayerNorm(self.output_dim)

    def forward(self, x, te, mask):
        B, N, TF = x.shape

        x = x.view(B, N, self.sample_len, -1)
        x = torch.concat((x, mask.view(B, N, self.sample_len, -1)), dim=-1)
        x = x.mean(dim=1)  # (B, T, F*2)

        state = x.view(B, 1, -1)
        state = torch.concat((state, te[:, -1:, :]), dim=-1)
        state = self.fc_state(state)

        grad = (x[:, 1:, :] - x[:, :-1, :]).view(B, 1, -1)
        grad = torch.concat((grad, te[:, -1:, :]), dim=-1)
        grad = self.fc_grad(grad)

        out = torch.concat((state, grad), dim=1)
        out = self.ln(out)
        return out


class SpatialTokenizer(nn.Module):
    """Convert per-node historical data into spatial tokens.

    Combines: fc1(x + mask) + mask_token(mask) + state_fc(time_emb + node_emb)
    Input: x (B, N, T*F), te (B, T, tim_dim), ne (N, node_emb_dim), mask (B, T, N, F)
    Output: (B, N, output_dim)
    """

    def __init__(self, sample_len, features, node_emb_dim, emb_dim, tim_dim, dropout,
                 use_node_embedding, output_dim=None):
        super().__init__()

        in_features = sample_len * features * 2  # data + mask
        self.use_node_embedding = use_node_embedding
        self.output_dim = output_dim if output_dim is not None else emb_dim

        state_features = tim_dim
        if use_node_embedding:
            state_features += node_emb_dim

        self.fc1 = nn.Sequential(
            nn.Linear(in_features, self.output_dim),
        )

        hidden_size = node_emb_dim
        self.state_fc = nn.Sequential(
            nn.Linear(state_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.output_dim),
        )

        self.mask_token = nn.Linear(sample_len * features, self.output_dim)
        self.ln = nn.LayerNorm(self.output_dim)

    def forward(self, x, te, ne, mask):
        # te: (B, T, tim_dim), ne: (N, node_emb_dim), mask: (B, T, N, F)
        B, N, TF = x.shape

        mask = mask.permute(0, 2, 1, 3).contiguous().view(B, N, -1)  # (B, N, T*F)
        x = torch.concat((x, mask), dim=-1)  # (B, N, T*F*2)
        x = self.fc1(x)  # (B, N, output_dim)

        state = te[:, -1:, :].repeat(1, N, 1)
        if self.use_node_embedding:
            ne = ne.unsqueeze(0).repeat(B, 1, 1)
            state = torch.concat((state, ne), dim=-1)
        state = self.state_fc(state)

        x += self.mask_token(mask)
        out = state + x
        out = self.ln(out)
        return out


class BiCAPForecaster(nn.Module):
    def __init__(self, basemodel, sample_len, output_len,
                 input_dim, output_dim,
                 node_emb_dim, latent_dim, num_latents,
                 adj_mx=None, dis_mx=None, use_node_embedding=True,
                 use_timetoken=True, use_spatial_attn=True,
                 dropout=0, trunc_k=16, t_dim=64, wo_conloss=False,
                 use_graph_transformer=False, graph_transformer_heads=6, graph_transformer_layers=1,
                 use_bidirectional=False, use_gating=True,
                 use_adaptive=False,
                 use_channel_attn=False, channel_attn_heads=4,
                 use_recon_loss=False, use_ncut_loss=False,
                 recon_weight=1.0, ncut_weight=1.0):
        super().__init__()

        self.topological_sort_node = True

        tim_dim = t_dim * 2  # day_of_5min + day_of_week

        self.setadj(adj_mx, dis_mx)

        self.output_dim = output_dim
        self.input_dim = input_dim

        self.emb_dim = basemodel.emb_dim
        self.basemodel = basemodel

        self.sample_len = sample_len
        self.output_len = output_len
        self.num_latents = num_latents

        # Graph Transformer or BiCAP spatial attention
        self.use_graph_transformer = use_graph_transformer
        self.use_spatial_attn = use_spatial_attn and not use_graph_transformer

        # LLM hidden size (2048 for LLaMA, 768 for GPT-2)
        llm_hidden_size = getattr(basemodel, 'llama_hidden_size', None)
        self.token_dim = llm_hidden_size if llm_hidden_size is not None else self.emb_dim
        if llm_hidden_size is not None:
            print(f"  Direct {self.token_dim}-dim pipeline (Tokenizers -> BiCAP -> LLaMA)")

        if use_graph_transformer:
            print("Using Graph Transformer (replacing BiCAP)")
            self.st_graph_transformer = SpatialTemporalGraphTransformer(
                d_model=self.emb_dim,
                num_heads=graph_transformer_heads,
                gcn_layers=graph_transformer_layers,
                dropout=dropout,
                use_sparse=True
            )
            if adj_mx is not None:
                self.adj_matrix_sparse = convert_adj_to_sparse(
                    torch.tensor(adj_mx, dtype=torch.float32),
                    threshold=0.0
                )
            else:
                self.adj_matrix_sparse = None
        elif self.use_spatial_attn:
            feat_tags = []
            if use_adaptive:
                feat_tags.append("Adaptive")
            if use_bidirectional:
                feat_tags.append("BiXT")
            if use_channel_attn:
                feat_tags.append("ChannelAttn")

            if feat_tags:
                print(f"Using BiCAP spatial attention with {' + '.join(feat_tags)}")
            else:
                print("Using BiCAP spatial attention (base)")
            self.wo_conloss = wo_conloss
            self.use_recon_loss = use_recon_loss
            self.use_ncut_loss = use_ncut_loss
            self.recon_weight = recon_weight
            self.ncut_weight = ncut_weight
            self.use_adaptive = use_adaptive
            self.spatial_attn = BCSPerceiver(
                latent_dim=latent_dim,
                num_latents=num_latents,
                emb_dim=self.emb_dim,
                sample_len=sample_len,
                features=input_dim,
                dropout=dropout,
                use_bidirectional=use_bidirectional,
                use_gating=use_gating,
                use_adaptive=use_adaptive,
                tim_dim=tim_dim,
                use_channel_attn=use_channel_attn,
                channel_attn_heads=channel_attn_heads,
                llm_hidden_size=self.token_dim,
                input_dim=self.token_dim
            )

        self.spatialTokenizer = SpatialTokenizer(
            sample_len=sample_len,
            features=input_dim,
            node_emb_dim=node_emb_dim,
            emb_dim=self.emb_dim,
            tim_dim=tim_dim,
            dropout=dropout,
            use_node_embedding=use_node_embedding,
            output_dim=self.token_dim
        )

        self.out_mlp = PredictionHead(
            input_dim=output_dim*sample_len,
            emb_dim=self.token_dim,
            output_dim=output_dim*output_len
        )

        self.timeembedding = TemporalEncoder(t_dim=t_dim)

        self.use_node_embedding = use_node_embedding
        if use_node_embedding:
            self.node_embd_layer = SpectralNodeEncoder(adj_mx=adj_mx, node_emb_dim=node_emb_dim, k=trunc_k, dropout=dropout)

        self.use_timetoken = use_timetoken
        if use_timetoken:
            self.timeTokenizer = TemporalTokenizer(
                sample_len=sample_len,
                features=input_dim,
                emb_dim=self.emb_dim,
                tim_dim=tim_dim,
                dropout=dropout,
                output_dim=self.token_dim
            )

        self.layer_norm = nn.LayerNorm(self.token_dim)


    def forward(self,x:torch.FloatTensor,timestamp:torch.Tensor,prompt_prefix:Optional[torch.LongTensor],mask:torch.LongTensor,text_prompts:Optional[list]=None):
        other_loss = []

        # timestamp (B,T,4)
        timestamp = timestamp[:,:self.sample_len,:]

        B,N,TF = x.shape #(Batch,N,T*features)
        # emb of time
        te = self.timeembedding(timestamp) #(B,T,tim_dim)
        # emb of nodes
        if self.use_node_embedding:
            ne = self.node_embd_layer()
        else:
            ne = None

        # spatial token
        spatial_token = self.spatialTokenizer(x,te,ne,mask)
        if self.topological_sort_node:
            spatial_token = spatial_token[:,self.node_order,:]

        # Spatial modeling: Graph Transformer or BiCAP
        st_embedding = spatial_token
        s_num = N

        if self.use_graph_transformer:
            spatial_token_expanded = spatial_token.unsqueeze(1)  # (B, 1, N, D)

            if self.adj_matrix_sparse is not None:
                adj_to_use = self.adj_matrix_sparse.to(spatial_token.device)
            else:
                adj_to_use = torch.eye(N).to(spatial_token.device)

            st_enhanced = self.st_graph_transformer(spatial_token_expanded, adj_to_use)
            st_embedding = st_enhanced.squeeze(1)  # (B, N, D)

        elif self.use_spatial_attn:
            s_num = self.num_latents
            te_for_adaptive = te if hasattr(self, 'use_adaptive') and self.use_adaptive else None
            spatial_token_before_compress = st_embedding  # save for reconstruction loss
            st_embedding, attn_weights = self.spatial_attn.encode(st_embedding, te=te_for_adaptive)
            if not self.wo_conloss:
                scale = attn_weights.sum(dim=1)

                adj_score = torch.einsum('bmn,bhn->bhm', self.adj_mx[None,:,:], attn_weights)
                other_loss.append(-((adj_score*attn_weights-attn_weights*attn_weights)).sum(dim=2).mean()*10)

                Dirichlet = torch.distributions.dirichlet.Dirichlet(self.alpha)
                other_loss.append(-Dirichlet.log_prob(torch.softmax(scale,dim=-1)).sum())

            # Reconstruction loss: compressed tokens should be able to reconstruct original node embeddings
            if self.use_recon_loss:
                # st_embedding: (B, M, D), attn_weights: (B, M, N)
                # Soft reconstruction: S^T @ compressed → (B, N, D)
                reconstructed = torch.bmm(attn_weights.transpose(1, 2), st_embedding)  # (B, N, D)
                loss_recon = 1.0 - nn.functional.cosine_similarity(
                    reconstructed, spatial_token_before_compress.detach(), dim=-1
                ).mean()
                other_loss.append(self.recon_weight * loss_recon)

            # Normalized Cut loss: encourage meaningful graph partitioning
            if self.use_ncut_loss:
                # S: (B, M, N) attention weights, adj_mx: (N, N)
                S = attn_weights
                # Assoc(m) = S_m A S_m^T  (intra-region connectivity)
                assoc = torch.einsum('bmn,nj,bmj->bm', S, self.adj_mx, S)  # (B, M)
                # Vol(m) = S_m D  (region volume)
                vol = torch.einsum('bmn,n->bm', S, self.d_mx)  # (B, M)
                loss_ncut = -(assoc / (vol + 1e-6)).mean()
                other_loss.append(self.ncut_weight * loss_ncut)

        if self.use_timetoken:
            time_tokens = self.timeTokenizer(x,te,mask)
            time_tokens_idx = st_embedding.shape[1]
            # 方案B: time_tokens 已經是 token_dim，無需投影
            st_embedding = torch.concat([time_tokens,st_embedding],dim=1)

        if prompt_prefix is not None:
            prompt_len,_ = prompt_prefix.shape
            prompt_embedding = self.basemodel.get_embedding(prompt_prefix).view(1,prompt_len,-1)
            prompt_embedding = prompt_embedding.repeat(B,1,1)
            st_embedding = torch.concat([prompt_embedding,st_embedding],dim=1)

        hidden_state = st_embedding

        hidden_state = self.basemodel(hidden_state,  text_prompts=text_prompts)
        s_state = hidden_state[:,-s_num:,:]

        if self.use_graph_transformer:
            s_state += spatial_token
        elif self.use_spatial_attn:
            s_state = self.spatial_attn.decode(s_state, spatial_token)
            s_state += spatial_token
        else:
            s_state += spatial_token

        if self.topological_sort_node:
            s_state = s_state[:,self.node_order_rev,:]

        if self.use_timetoken:
            t_state = hidden_state[:,-time_tokens_idx-1:-time_tokens_idx,:]
            t_state += time_tokens[:,-1:,:]
            # 方案B: 所有維度一致 (token_dim)，無需投影
            s_state += t_state

        s_state = self.layer_norm(s_state)

        out = self.out_mlp(s_state)

        return out, other_loss

    def grad_state_dict(self):
        params_to_save = filter(lambda p: p[1].requires_grad, self.named_parameters())
        save_list = [p[0] for p in params_to_save]
        return  {name: param.detach() for name, param in self.state_dict().items() if name in save_list}
        
    
    def save(self, path:str):
        
        selected_state_dict = self.grad_state_dict()
        torch.save(selected_state_dict, path)
    
    def load(self, path:str):

        loaded_params = torch.load(path)
        self.load_state_dict(loaded_params,strict=False)
    
    def params_num(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_params += sum(p.numel() for p in self.buffers())
        
        total_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        
        return total_params, total_trainable_params

    def setadj(self,adj_mx,dis_mx):

        self.adj_mx = torch.tensor(adj_mx).cuda()
        self.dis_mx = torch.tensor(dis_mx).cuda()
        self.d_mx = self.adj_mx.sum(dim=1)
        N = self.adj_mx.shape[0]
        self.alpha = torch.tensor([1.05] * N).cuda() + torch.softmax(self.d_mx,dim=0)*5 
        self.node_order,self.node_order_rev = compute_node_ordering(adj_mx)
