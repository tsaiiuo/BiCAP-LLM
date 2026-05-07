import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from core.sinusoidal_pe import SinusoidalPositionEncoder

class BCSPerceiver(nn.Module):
    """
    Bi-directional Cross-attention Spatial Perceiver (BCSPerceiver) for spatial compression.

    Compresses N input nodes into a smaller set of latent tokens, processes them
    through an LLM backbone, then expands back to N nodes.

    Processing pipeline:
    1. [Adaptive] Adaptive latent tokens: base + spatial_context * temporal_scale (when enabled)
    2. [BiXT] Backward attention: input nodes attend to latent tokens
    3. [BiXT] Enhanced forward attention: latent tokens attend to backward-enhanced nodes
    4. [BiXT] Gating: blend original forward and enhanced forward outputs
    5. [Channel] Channel-wise attention across latent tokens (when enabled)
    6. Output projection to llm_hidden_size

    Draws from:
    - BiXT (NeurIPS 2024) - Bi-directional cross-attention mechanism
    - STAEformer (CIKM 2023) - Adaptive embedding strategy
    - iTransformer (ICLR 2024) - Inverted attention for multivariate time series

    Args:
        latent_dim: Bottleneck dimension for Q/K projection
        num_latents: Number of compressed latent tokens
        emb_dim: Input embedding dimension (768 for GPT-2)
        sample_len: Time sequence length
        features: Number of input features
        dropout: Dropout rate
        use_bidirectional: Toggle BiXT-style bidirectional cross-attention
        use_gating: Toggle gating mechanism for fusion (only with bidirectional)
        use_adaptive: Toggle adaptive latent tokens (STAEformer-inspired)
        tim_dim: Time embedding dimension (for temporal adapter, default 128)
        use_channel_attn: Toggle channel-wise attention (iTransformer-inspired)
        channel_attn_heads: Number of attention heads for channel attention (default 4)
        llm_hidden_size: LLM hidden dimension (default None = use emb_dim)
    """
    def __init__(self, latent_dim, num_latents, emb_dim, sample_len, features, dropout,
                 use_bidirectional=False, use_gating=True,
                 use_adaptive=False, tim_dim=128,
                 use_channel_attn=False, channel_attn_heads=4,
                 llm_hidden_size=None, input_dim=None):
        super().__init__()

        self.num_latents = num_latents
        self.num_heads = 4
        self.latent_dim = latent_dim
        self.use_bidirectional = use_bidirectional
        self.use_gating = use_gating
        self.emb_dim = emb_dim
        self.use_adaptive = use_adaptive
        self.use_channel_attn = use_channel_attn

        # input_dim: input dimension (LLaMA uses 2048, GPT-2 uses 768)
        self.input_dim = input_dim if input_dim is not None else emb_dim

        # llm_hidden_size: output dimension (defaults to input_dim)
        self.llm_hidden_size = llm_hidden_size if llm_hidden_size is not None else self.input_dim

        self.latent_tokens = nn.Parameter(torch.randn(1, num_latents, latent_dim))
        self.pe = SinusoidalPositionEncoder(d_model=latent_dim, dropout=dropout, max_len=1024)

        # ========= Adaptive Latent Tokens (STAEformer-inspired) =========
        if use_adaptive:
            self.spatial_adapter = nn.Sequential(
                nn.Linear(self.input_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, num_latents)
            )
            self.spatial_value_fc = nn.Linear(self.input_dim, latent_dim)

            self.temporal_adapter = nn.Sequential(
                nn.Linear(tim_dim, latent_dim),
                nn.Tanh()
            )

            self.adaptive_ln = nn.LayerNorm(latent_dim)
        # ==================================================================

        # Forward attention: latent tokens attend to input nodes
        self.enc_mha = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=self.num_heads, batch_first=True, dropout=dropout)
        # Decode attention: vdim = llm_hidden_size to accept LLM output directly
        self.dec_mha = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=self.num_heads, batch_first=True, dropout=dropout, vdim=self.llm_hidden_size)

        # ========= BiXT: Backward attention path =========
        if use_bidirectional:
            self.backward_mha = nn.MultiheadAttention(
                embed_dim=latent_dim,
                num_heads=self.num_heads,
                batch_first=True,
                dropout=dropout
            )
            self.backward_ln = nn.LayerNorm(latent_dim)

            if use_gating:
                self.gate_fc = nn.Sequential(
                    nn.Linear(latent_dim * 2, latent_dim),
                    nn.Sigmoid()
                )
        # ==================================================

        # ========= Channel-wise Attention (iTransformer-inspired) =========
        if use_channel_attn:
            self.channel_attn = nn.MultiheadAttention(
                embed_dim=num_latents,
                num_heads=channel_attn_heads,
                batch_first=True,
                dropout=dropout
            )
            self.channel_ln = nn.LayerNorm(latent_dim)
        # =====================================================================

        # Encode output: project to LLM hidden size
        self.enc_fc = nn.Linear(in_features=latent_dim, out_features=self.llm_hidden_size)
        # Decode output: project back to input dimension
        self.dec_fc = nn.Linear(in_features=latent_dim, out_features=self.input_dim)

        # Input projection: input_dim -> latent_dim
        self.x_fc = nn.Linear(in_features=self.input_dim, out_features=latent_dim)

        self.en_ln = nn.LayerNorm(self.llm_hidden_size)
        self.de_ln = nn.LayerNorm(self.input_dim)


    def encode(self, x, te=None):
        """
        Compress input nodes into latent representation.

        Args:
            x: (B, N, D) input node features
            te: (B, T, tim_dim) time embedding (optional, for adaptive mode)

        Returns:
            out: (B, num_latents, llm_hidden_size) compressed representation
            attn_weights: (B, num_latents, N) attention weights
        """
        B, N, H = x.shape

        kv = self.x_fc(x)  # (B, N, latent_dim)

        # ========= Adaptive Latent Tokens =========
        if self.use_adaptive and te is not None:
            spatial_logits = self.spatial_adapter(x)  # (B, N, num_latents)
            spatial_weights = F.softmax(spatial_logits, dim=1).transpose(1, 2)  # (B, num_latents, N)

            spatial_values = self.spatial_value_fc(x)  # (B, N, latent_dim)
            spatial_context = torch.bmm(spatial_weights, spatial_values)  # (B, num_latents, latent_dim)

            temporal_scale = self.temporal_adapter(te[:, -1, :]).unsqueeze(1)  # (B, 1, latent_dim)

            adaptive_latent = self.latent_tokens + spatial_context * (1 + temporal_scale)
            adaptive_latent = self.adaptive_ln(adaptive_latent)

            q = self.pe(adaptive_latent)  # (B, num_latents, latent_dim)
        else:
            q = self.pe(self.latent_tokens)  # (1, num_latents, latent_dim)
        # =====================================================

        # ========= Forward Attention =========
        q_batched = q if q.shape[0] == B else q.repeat(B, 1, 1)

        forward_out, attn_weights = self.enc_mha(
            query=q_batched,
            key=self.pe(kv),
            value=kv
        )  # (B, num_latents, latent_dim)

        if self.use_bidirectional:
            # ========= BiXT: Backward Attention =========
            backward_out, _ = self.backward_mha(
                query=self.pe(kv),
                key=q_batched,
                value=q_batched
            )  # (B, N, latent_dim)

            enhanced_kv = self.backward_ln(kv + backward_out)  # (B, N, latent_dim)

            # ========= Enhanced Forward Attention =========
            enhanced_forward_out, enhanced_attn_weights = self.enc_mha(
                query=q_batched,
                key=self.pe(enhanced_kv),
                value=enhanced_kv
            )  # (B, num_latents, latent_dim)

            if self.use_gating:
                gate = self.gate_fc(torch.cat([forward_out, enhanced_forward_out], dim=-1))
                out = gate * forward_out + (1 - gate) * enhanced_forward_out
                attn_weights = enhanced_attn_weights
            else:
                out = enhanced_forward_out
                attn_weights = enhanced_attn_weights
        else:
            out = forward_out

        # ========= Channel-wise Attention =========
        if self.use_channel_attn:
            out_t = out.transpose(1, 2)  # (B, latent_dim, num_latents)
            channel_out, _ = self.channel_attn(out_t, out_t, out_t)
            channel_out = channel_out.transpose(1, 2)  # (B, num_latents, latent_dim)
            out = self.channel_ln(out + channel_out)
        # ==============================================

        out = self.enc_fc(out)
        out = self.en_ln(out)

        return out, attn_weights

    def decode(self, hidden_state, x):
        """
        Expand LLM output back to original node dimension.

        Args:
            hidden_state: (B, num_latents, llm_hidden_size) LLM output
            x: (B, N, input_dim) original node features (for query construction)

        Returns:
            out: (B, N, input_dim) decoded node features
        """
        B, _, _ = hidden_state.shape

        q = self.pe(self.x_fc(x))
        k = self.pe(self.latent_tokens)
        v = hidden_state

        out, _ = self.dec_mha(query=q, key=k.repeat(B, 1, 1), value=v)  # (B, N, latent_dim)

        out = self.dec_fc(out)
        out = self.de_ln(out)

        return out
