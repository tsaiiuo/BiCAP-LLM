"""
Spatial-Temporal Graph Transformer
Inspired by STGformer (2024)

Alternative spatial-temporal modeling approach using graph-based processing:
- GraphConvolution: Captures road network topology
- Spatial Attention: Models global spatial dependencies
- Temporal Attention: Captures temporal patterns

Design choices:
- Single GCN layer (following STGformer)
- Sparse adjacency matrix support for efficiency
- Reduced attention heads (6 instead of 12)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GraphConvolution(nn.Module):
    """
    Graph Convolution layer with optional sparse support.

    Computes: D^{-1/2} A D^{-1/2} X W + b
    where A is the adjacency matrix, X is node features, W is weights.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with Xavier uniform."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (B, N, D_in)
            adj: Adjacency matrix (N, N) - can be sparse or dense

        Returns:
            output: (B, N, D_out)
        """
        B, N, D_in = x.shape

        # Save original dtype for FP16 compatibility
        original_dtype = x.dtype

        # Step 1: X @ W (feature transformation)
        # Convert weight to match input dtype (for FP16 support)
        weight = self.weight.to(x.dtype)
        support = torch.matmul(x, weight)  # (B, N, D_out)

        # Step 2: Normalize adjacency if not already normalized
        if not hasattr(self, 'adj_normalized') or self.adj_normalized.shape != adj.shape:
            self.adj_normalized = self._normalize_adj(adj)

        # Step 3: Graph aggregation
        if adj.is_sparse:
            # Sparse matrix multiplication
            # NOTE: torch.sparse.mm doesn't support FP16, so autocast must be disabled
            from torch.cuda.amp import autocast as amp_autocast

            output = []
            for b in range(B):
                # Disable autocast for sparse operations
                with amp_autocast(enabled=False):
                    # Convert support to float32 for sparse operations
                    support_fp32 = support[b].float()

                    # Ensure adjacency is float32
                    if self.adj_normalized.dtype != torch.float32:
                        indices = self.adj_normalized._indices()
                        values = self.adj_normalized._values().float()
                        size = self.adj_normalized.size()
                        self.adj_normalized = torch.sparse_coo_tensor(
                            indices, values, size, dtype=torch.float32, device=self.adj_normalized.device
                        ).coalesce()

                    # Sparse MM (completely outside autocast)
                    out_b = torch.sparse.mm(self.adj_normalized, support_fp32)  # (N, D_out)

                # Convert back to original dtype if needed
                if original_dtype == torch.float16:
                    out_b = out_b.half()

                output.append(out_b)
            output = torch.stack(output, dim=0)  # (B, N, D_out)
        else:
            # Dense matrix multiplication
            output = torch.matmul(self.adj_normalized, support)  # (B, N, D_out)

        # Step 4: Add bias
        if self.bias is not None:
            bias = self.bias.to(output.dtype)
            output = output + bias

        return output

    def _normalize_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Symmetric normalization of adjacency matrix: D^{-1/2} A D^{-1/2}

        Args:
            adj: (N, N) adjacency matrix

        Returns:
            Normalized adjacency matrix (always float32 for sparse)
        """
        if adj.is_sparse:
            # Sparse adjacency - always keep in float32 for sparse operations
            adj_dense = adj.to_dense().float()  # Force float32
            degree = adj_dense.sum(1)
            degree_inv_sqrt = torch.pow(degree, -0.5)
            degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.

            # D^{-1/2}
            degree_mat_inv_sqrt = torch.diag(degree_inv_sqrt)

            # D^{-1/2} A D^{-1/2}
            adj_normalized = degree_mat_inv_sqrt @ adj_dense @ degree_mat_inv_sqrt

            # Convert back to sparse and keep as float32
            return adj_normalized.to_sparse().float()
        else:
            # Dense adjacency
            degree = adj.sum(1)
            degree_inv_sqrt = torch.pow(degree, -0.5)
            degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.

            # Normalize: D^{-1/2} A D^{-1/2}
            adj_normalized = degree_inv_sqrt.view(-1, 1) * adj * degree_inv_sqrt.view(1, -1)

            return adj_normalized


class SpatialTemporalGraphTransformer(nn.Module):
    """
    Unified Spatial-Temporal Graph Transformer.

    Architecture:
        1. GCN: Local spatial patterns from road network topology
        2. Spatial Attention: Global spatial dependencies
        3. Temporal Attention: Temporal patterns across timesteps

    Design inspired by STGformer (2024):
        - Single-layer design (no deep stacking)
        - Unified spatial-temporal processing
        - Efficient for short-term forecasting
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 6,
        gcn_layers: int = 1,
        dropout: float = 0.1,
        use_sparse: bool = True
    ):
        """
        Args:
            d_model: Embedding dimension (768 for GPT-2)
            num_heads: Number of attention heads (6 for efficiency)
            gcn_layers: Number of GCN layers (1 recommended)
            dropout: Dropout rate
            use_sparse: Whether to use sparse adjacency matrix
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.gcn_layers = gcn_layers
        self.use_sparse = use_sparse

        # Branch 1: Graph Convolution (local spatial patterns)
        self.gcn = nn.ModuleList([
            GraphConvolution(d_model, d_model)
            for _ in range(gcn_layers)
        ])
        self.gcn_activation = nn.ReLU()
        self.gcn_dropout = nn.Dropout(dropout)

        # Branch 2: Spatial Attention (global spatial dependencies)
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.spatial_norm = nn.LayerNorm(d_model)

        # Fusion of GCN + Spatial Attention outputs
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        self.fusion_norm = nn.LayerNorm(d_model)

        # Temporal Attention (across time dimension)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(d_model)

        print(f"Graph Transformer initialized:")
        print(f"   - d_model: {d_model}")
        print(f"   - num_heads: {num_heads}")
        print(f"   - gcn_layers: {gcn_layers}")
        print(f"   - use_sparse: {use_sparse}")

    def forward(
        self,
        x: torch.Tensor,
        adj_matrix: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input features (B, T, N, D) or (B, N, T, D)
                - B: batch size
                - T: number of timesteps
                - N: number of nodes (307)
                - D: feature dimension (768)
            adj_matrix: Adjacency matrix (N, N)
            mask: Optional attention mask

        Returns:
            output: Enhanced features (B, T, N, D)
        """
        # Handle both (B, T, N, D) and (B, N, T, D) inputs
        if x.dim() == 4:
            B, dim1, dim2, D = x.shape
            # Assume if dim1 < dim2, it's (B, T, N, D), otherwise (B, N, T, D)
            if dim1 < dim2:
                # Input is (B, T, N, D) - correct format
                T, N = dim1, dim2
            else:
                # Input is (B, N, T, D) - need to transpose
                x = x.transpose(1, 2)  # (B, T, N, D)
                T, N = dim2, dim1
        else:
            raise ValueError(f"Expected 4D input, got {x.dim()}D")

        # ========= Stage 1: Spatial Modeling (GCN + Attention) =========

        # Branch 1: GCN (local topology)
        gcn_out = x.clone()
        for t in range(T):
            x_t = x[:, t, :, :]  # (B, N, D)

            # Apply GCN layers
            for gcn_layer in self.gcn:
                x_t = gcn_layer(x_t, adj_matrix)  # (B, N, D)
                x_t = self.gcn_activation(x_t)
                x_t = self.gcn_dropout(x_t)

            gcn_out[:, t, :, :] = x_t

        # Branch 2: Spatial Attention (global dependencies)
        spatial_out = []
        for t in range(T):
            x_t = x[:, t, :, :]  # (B, N, D)

            # Self-attention across nodes (N dimension)
            attn_out, _ = self.spatial_attn(x_t, x_t, x_t, attn_mask=mask)  # (B, N, D)
            attn_out = self.spatial_norm(x_t + attn_out)  # Residual connection

            spatial_out.append(attn_out)

        spatial_out = torch.stack(spatial_out, dim=1)  # (B, T, N, D)

        # Fuse GCN + Spatial Attention
        combined = torch.cat([gcn_out, spatial_out], dim=-1)  # (B, T, N, 2D)
        x_spatial = self.fusion(combined)  # (B, T, N, D)
        x_spatial = self.fusion_norm(x + x_spatial)  # Residual connection

        # ========= Stage 2: Temporal Modeling =========

        temporal_out = []
        for n in range(N):
            x_n = x_spatial[:, :, n, :]  # (B, T, D)

            # Self-attention across time (T dimension)
            attn_out, _ = self.temporal_attn(x_n, x_n, x_n)  # (B, T, D)
            attn_out = self.temporal_norm(x_n + attn_out)  # Residual connection

            temporal_out.append(attn_out)

        temporal_out = torch.stack(temporal_out, dim=2)  # (B, T, N, D)

        return temporal_out


def convert_adj_to_sparse(adj_dense: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """
    Convert dense adjacency matrix to sparse COO format.

    Args:
        adj_dense: Dense adjacency matrix (N, N)
        threshold: Remove edges with weight below this threshold

    Returns:
        Sparse adjacency matrix in COO format
    """
    # Remove weak edges
    if threshold > 0:
        adj_dense = adj_dense.clone()
        adj_dense[adj_dense < threshold] = 0

    # Convert to sparse COO format
    adj_sparse = adj_dense.to_sparse_coo()

    return adj_sparse


# ========= Unit Test =========

if __name__ == "__main__":
    print("Testing Graph Transformer...")

    # Test configuration
    B, T, N, D = 4, 12, 307, 768
    num_heads = 6

    # Create dummy data
    x = torch.randn(B, T, N, D)
    adj_matrix = torch.rand(N, N)

    # Normalize adjacency (simple version)
    adj_matrix = (adj_matrix + adj_matrix.T) / 2  # Make symmetric
    adj_matrix = (adj_matrix > 0.5).float()  # Binarize

    print(f"\nInput shape: {x.shape}")
    print(f"Adjacency shape: {adj_matrix.shape}")
    print(f"Adjacency density: {adj_matrix.sum() / (N*N):.2%}")

    # Test GraphConvolution
    print("\n--- Testing GraphConvolution ---")
    gcn = GraphConvolution(D, D)
    x_test = x[:, 0, :, :]  # (B, N, D)
    gcn_out = gcn(x_test, adj_matrix)
    print(f"GCN output shape: {gcn_out.shape}")
    assert gcn_out.shape == (B, N, D), "GCN output shape mismatch!"
    print("GraphConvolution test passed!")

    # Test SpatialTemporalGraphTransformer
    print("\n--- Testing SpatialTemporalGraphTransformer ---")
    model = SpatialTemporalGraphTransformer(
        d_model=D,
        num_heads=num_heads,
        gcn_layers=1,
        use_sparse=False
    )

    output = model(x, adj_matrix)
    print(f"Output shape: {output.shape}")
    assert output.shape == (B, T, N, D), "Output shape mismatch!"
    print("SpatialTemporalGraphTransformer test passed!")

    # Test with sparse adjacency
    print("\n--- Testing with Sparse Adjacency ---")
    adj_sparse = convert_adj_to_sparse(adj_matrix, threshold=0.5)
    print(f"Sparse adjacency: {adj_sparse.shape}, nnz={adj_sparse._nnz()}")

    model_sparse = SpatialTemporalGraphTransformer(
        d_model=D,
        num_heads=num_heads,
        gcn_layers=1,
        use_sparse=True
    )

    output_sparse = model_sparse(x, adj_sparse)
    print(f"Output shape (sparse): {output_sparse.shape}")
    assert output_sparse.shape == (B, T, N, D), "Sparse output shape mismatch!"
    print("Sparse adjacency test passed!")

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
