import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torchcde
from typing import Callable, Dict, Optional


def ensure_directory(path: str, create: bool = False) -> bool:
    """Check if a directory exists, optionally creating it."""
    if os.path.exists(path):
        return True
    if create:
        os.mkdir(path)
        return True
    return False


def log_config(config: Dict, logger: Optional[logging.Logger] = None) -> None:
    """Pretty-print a nested configuration dictionary."""
    writer = logger.info if logger is not None else print
    writer('*' * 10 + 'config' + '*' * 10)
    for section in config.keys():
        writer(f'[{section}]')
        entries = config[section]
        for key in entries:
            writer(f'{key} = {entries[key]}\n')
    writer('*' * 26)


def timestamp_str() -> str:
    """Return the current local time as a formatted string."""
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))


def initialize_weights(
    model: nn.Module,
    filter_fn: Optional[Callable[[nn.Parameter], bool]] = None,
) -> None:
    """Apply Xavier uniform (2-D+) or uniform (1-D) init to model parameters."""
    for param in model.parameters():
        if filter_fn is not None and not filter_fn(param):
            continue
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.uniform_(param)


def random_mask(
    observed_mask: torch.Tensor,
    min_miss_ratio: float = 0.,
    max_miss_ratio: float = 1.,
) -> torch.Tensor:
    """Generate a random conditional mask by dropping a random fraction of observed entries."""
    rand_vals = torch.rand_like(observed_mask) * observed_mask
    rand_vals = rand_vals.reshape(-1)

    ratio = np.random.rand()
    ratio = ratio * (max_miss_ratio - min_miss_ratio) + min_miss_ratio

    num_observed = observed_mask.sum().item()
    num_to_mask = round(num_observed * ratio)
    rand_vals[rand_vals.topk(num_to_mask).indices] = -1

    cond_mask = (rand_vals > 0).reshape(observed_mask.shape).float()
    return cond_mask


def block_mask(
    observed_mask: torch.Tensor,
    target_strategy: str = 'hybrid',
    min_seq: int = 3,
    max_seq: int = 12,
) -> torch.Tensor:
    """Generate a block-structured conditional mask for imputation tasks."""
    rand_sensor = torch.rand_like(observed_mask)
    randint = np.random.randint
    ratio = np.random.rand() * 0.15
    selected = rand_sensor < ratio

    for col in range(observed_mask.shape[1]):
        indices = np.flatnonzero(selected[:, col])
        if not len(indices):
            continue
        fault_len = min_seq
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq))
        extended = np.concatenate([np.arange(i, i + fault_len) for i in indices])
        extended = np.unique(extended)
        extended = np.clip(extended, 0, observed_mask.shape[0] - 1)
        selected[extended, col] = True

    base_noise = torch.rand_like(observed_mask) < 0.05
    combined = selected | base_noise
    blk_mask = 1 - combined.to(torch.float32)

    cond_mask = observed_mask.clone()
    choice = np.random.rand()
    if target_strategy == "hybrid" and choice > 0.7:
        cond_mask = random_mask(observed_mask, 0., 1.)
    else:
        cond_mask = blk_mask * cond_mask

    return cond_mask


def linear_interpolation(data: torch.Tensor) -> torch.Tensor:
    """Compute linear interpolation coefficients for continuous-time models."""
    return torchcde.linear_interpolation_coeffs(data)
