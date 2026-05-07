import torch
from typing import List, Union


class ZScoreNormalizer:
    """Standardize features by removing the mean and scaling to unit variance."""

    def __init__(self, mean: List[float], std: List[float]) -> None:
        self.mean = mean
        self.std = std

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply z-score normalization per feature channel."""
        num_features = data.shape[-1]
        result = torch.zeros_like(data)
        for idx in range(num_features):
            result[..., idx:idx + 1] = (data[..., idx:idx + 1] - self.mean[idx]) / self.std[idx]
        return result

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Recover original scale from z-score normalized data."""
        num_features = data.shape[-1]
        result = torch.zeros_like(data)
        for idx in range(num_features):
            result[..., idx:idx + 1] = (data[..., idx:idx + 1] * self.std[idx]) + self.mean[idx]
        return result


class RangeNormalizer:
    """Scale features to the range [-1, 1] using min-max normalization."""

    def __init__(self, min_val: torch.Tensor, max_val: torch.Tensor) -> None:
        self._min = min_val
        self._max = max_val

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Map values to [-1, 1]."""
        data = 1. * (data - self._min) / (self._max - self._min)
        data = data * 2. - 1.
        return data

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Recover original scale from [-1, 1] normalized data."""
        num_features = data.shape[-1]
        data = (data + 1.) / 2.
        data = 1. * data * (self._max[..., :num_features] - self._min[..., :num_features]) + self._min[..., :num_features]
        return data
