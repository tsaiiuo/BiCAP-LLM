import torch
import torch.nn as nn


class ReversibleInstanceNorm(nn.Module):
    """Reversible instance normalization for non-stationary time series."""

    def __init__(self, num_features: int, eps: float = 1e-5,
                 affine: bool = True) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """Apply normalization (mode='norm') or denormalization (mode='denorm')."""
        if mode == 'norm':
            reduce_dims = tuple(range(1, x.ndim - 1))
            self.mean = torch.mean(x, dim=reduce_dims, keepdim=True).detach()
            self._std = torch.sqrt(
                torch.var(x, dim=reduce_dims, keepdim=True, unbiased=False) + self.eps
            ).detach()
            x = (x - self.mean) / self._std
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
            x = x * self._std + self.mean
        else:
            raise NotImplementedError(f"Unknown mode: {mode}")
        return x
