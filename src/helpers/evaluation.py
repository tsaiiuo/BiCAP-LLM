import torch
from typing import List, Optional, Tuple


def compute_mae(pred: torch.Tensor, target: torch.Tensor,
                mask_value: Optional[float] = None) -> torch.Tensor:
    """Mean absolute error with optional masking."""
    if mask_value is not None:
        valid = target > mask_value
        p, t = pred[valid], target[valid]
    else:
        p, t = pred, target
    return torch.mean(torch.abs(t - p))


def compute_mse(pred: torch.Tensor, target: torch.Tensor,
                mask_value: Optional[float] = None) -> torch.Tensor:
    """Mean squared error with optional masking."""
    if mask_value is not None:
        valid = target > mask_value
        p, t = pred[valid], target[valid]
    else:
        p, t = pred, target
    return torch.mean((p - t) ** 2)


def compute_rmse(pred: torch.Tensor, target: torch.Tensor,
                 mask_value: Optional[float] = None) -> torch.Tensor:
    """Root mean squared error with optional masking."""
    if mask_value is not None:
        valid = target > mask_value
        p, t = pred[valid], target[valid]
    else:
        p, t = pred, target
    return torch.sqrt(torch.mean((p - t) ** 2))


def compute_mape(pred: torch.Tensor, target: torch.Tensor,
                 mask_value: float = 1e-6) -> torch.Tensor:
    """Mean absolute percentage error with optional masking."""
    if mask_value is not None:
        valid = target > mask_value
        p, t = pred[valid], target[valid]
    else:
        p, t = pred, target
    return torch.mean(torch.abs(torch.div((t - p), t)))


def compute_mape_per_node(pred: torch.Tensor, target: torch.Tensor,
                          mask_value: float = 1e-6) -> torch.Tensor:
    """Per-node MAPE, returns a tensor of shape (...,) without the last dim."""
    if mask_value is not None:
        valid = target > mask_value
        pred = pred * valid
        target = target * valid + (1 - valid.float())
        count = valid.sum(dim=-1)
    return torch.sum(torch.abs(torch.div((target - pred) * valid, target)), dim=-1) / count


def evaluate_predictions(
    predicts: torch.Tensor,
    targets: torch.Tensor,
    eval_mask: torch.Tensor,
) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    """Compute MAE, RMSE, MAPE, MAPE@10, MAPE@20 per forecast horizon."""
    num_horizons = targets.shape[-1]

    mae: List[float] = []
    for h in range(num_horizons):
        m = eval_mask[..., h]
        mae.append(compute_mae(pred=predicts[..., h][m],
                                target=targets[..., h][m]).item())

    rmse: List[float] = []
    for h in range(num_horizons):
        m = eval_mask[..., h]
        rmse.append(compute_rmse(pred=predicts[..., h][m],
                                  target=targets[..., h][m]).item())

    mape: List[float] = []
    for h in range(num_horizons):
        m = eval_mask[..., h]
        mape.append(compute_mape(pred=predicts[..., h][m],
                                  target=targets[..., h][m]).item())

    mape_10: List[float] = []
    for h in range(num_horizons):
        m = eval_mask[..., h] & (targets[..., 0] >= 10)
        mape_10.append(compute_mape(pred=predicts[..., h][m],
                                     target=targets[..., h][m]).item())

    mape_20: List[float] = []
    for h in range(num_horizons):
        m = eval_mask[..., h] & (targets[..., 0] >= 20)
        mape_20.append(compute_mape(pred=predicts[..., h][m],
                                     target=targets[..., h][m]).item())

    return mae, rmse, mape, mape_10, mape_20
