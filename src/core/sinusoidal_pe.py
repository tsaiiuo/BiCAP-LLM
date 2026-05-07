import math
import torch
import torch.nn as nn


class SinusoidalPositionEncoder(nn.Module):
    """Fixed sinusoidal positional encoding following Vaswani et al."""

    def __init__(self, d_model: int, dropout: float,
                 max_len: int = 1000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            -torch.arange(0, d_model, 2, dtype=torch.float32) * (math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('P', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding and apply dropout."""
        x = x + self.P[:, :x.shape[1], :]
        return self.dropout(x)
