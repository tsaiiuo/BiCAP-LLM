"""Traffic data loading and dataset construction utilities.

Provides reader classes for various traffic datasets (PEMS, NYC Taxi, etc.)
and a sliding-window based dataset builder for spatial-temporal forecasting.
"""

from typing import Optional, Tuple, List, Dict, Callable

import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data

from pipeline.preprocessing import ZScoreNormalizer, RangeNormalizer
from helpers.graph_ops import (
    load_adjacency_from_csv,
    load_symmetric_adjacency,
    normalize_adjacency,
    compute_shortest_paths,
)
from helpers.graph_ops import load_adjacency_from_csv
from helpers.misc import random_mask, block_mask


def sliding_window_samples(
    data: torch.Tensor,
    sample_len: int,
    step: int = 1,
) -> torch.Tensor:
    """Extract fixed-length subsequences from a time-series tensor via sliding window.

    Args:
        data: Tensor of shape (T, ...) representing the full time series.
        sample_len: Length of each extracted subsequence.
        step: Stride between consecutive windows.

    Returns:
        Tensor of shape (num_samples, sample_len, ...).
    """
    sample: List[torch.Tensor] = []

    for i in range(0, data.shape[0] - sample_len, step):
        sample.append(torch.unsqueeze(data[i:i + sample_len], 0))

    if (data.shape[0] - sample_len) % step != 0:
        sample.append(torch.unsqueeze(data[-sample_len:], 0))

    sample = torch.concat(sample, dim=0)

    return sample


class TrafficDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapping pre-split traffic forecasting tensors.

    Attributes:
        history: Input observations, shape (B, sample_len, node_num, features).
        target: Ground-truth future values, shape (B, output_len, node_num, features).
        timestamp: Temporal features, shape (B, window_size, 5).
        cond_mask: Conditional mask for training, shape (B, sample_len, node_num, ...).
        ob_mask: Observation mask, shape (B, window_size, node_num, ...).
    """

    history: torch.Tensor
    target: torch.Tensor
    timestamp: torch.Tensor
    cond_mask: torch.Tensor
    ob_mask: torch.Tensor

    def __init__(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
        timestamp: torch.Tensor,
        cond_mask: torch.Tensor,
        ob_mask: torch.Tensor,
        training: bool = False,
    ) -> None:
        self.history = history
        self.target = target
        self.timestamp = timestamp
        self.cond_mask = cond_mask
        self.ob_mask = ob_mask
        self.training = training

    def __len__(self) -> int:
        return self.history.shape[0]

    def __getitem__(self, index: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        return (
            self.history[index],
            self.target[index],
            self.timestamp[index],
            self.cond_mask[index],
            self.ob_mask[index],
        )


class TrafficDataLoader:
    """Base loader that reads raw traffic data and produces train/val/test datasets.

    Subclasses must implement ``read_data`` to handle format-specific loading.
    After construction the following attributes are available:

    * ``data`` -- full tensor of shape (T, node_num, features)
    * ``node_num`` / ``features`` -- spatial and feature dimensions
    * ``adj_mx`` / ``distance_mx`` -- adjacency and distance matrices
    * ``timestamp`` -- temporal feature tensor
    * ``mask`` / ``mask_eval`` -- observation and evaluation masks
    """

    node_num: int
    features: int
    data: torch.Tensor
    timestamp: torch.Tensor
    mask: torch.Tensor
    mask_eval: torch.Tensor

    def __init__(
        self,
        data_path: str,
        adj_path: str,
        dataset: str,
        node_shuffle_seed: Optional[int] = None,
    ) -> None:
        self.dataset = dataset

        self.data, self.node_num, self.features, \
            self.adj_mx, self.distance_mx, \
            self.timestamp, self.mask, self.mask_eval = self.read_data(data_path, adj_path)

        if node_shuffle_seed is not None:
            rdm = np.random.RandomState(node_shuffle_seed)
            idx = np.arange(self.node_num)
            rdm.shuffle(idx)
            idx = torch.from_numpy(idx)
            self.data = self.data[:, idx, :]
            self.adj_mx = self.adj_mx[idx, :][:, idx]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_splits(
        self,
        sample_len: int,
        output_len: int,
        window_size: int,
        input_dim: int,
        output_dim: int,
        train_ratio: float,
        val_ratio: float,
        target_strategy: str,
        few_shot: float = 1,
    ) -> Tuple[TrafficDataset, TrafficDataset, TrafficDataset]:
        """Split the full dataset into train / validation / test TrafficDatasets.

        Args:
            sample_len: Number of historical time steps used as input.
            output_len: Number of future time steps to predict.
            window_size: Total window covering both input and output.
            input_dim: Number of feature channels to keep for input.
            output_dim: Number of feature channels to keep for target.
            train_ratio: Fraction of data reserved for training.
            val_ratio: Fraction of data reserved for validation.
            target_strategy: Masking strategy ('random' or 'hybrid').
            few_shot: Fraction of training data to actually use.

        Returns:
            A tuple of (train_dataset, val_dataset, test_dataset).
        """
        self.data = self.data.float().cuda()

        self.timestamp = self.timestamp.cuda()
        self.mask = self.mask.float().cuda()
        self.mask_eval = self.mask_eval.float().cuda()

        all_len = self.data.shape[0]
        train_len = int(all_len * train_ratio)
        val_len = int(all_len * val_ratio)

        train_range = [0, int(train_len * few_shot)]
        val_range = [train_len, train_len + val_len]
        test_range = [train_len + val_len, all_len]

        # Fit normalizer on training split only
        scaler_mask = self.mask[train_range[0]:train_range[1]] != 0
        scaler_data = self.data[train_range[0]:train_range[1]]
        dim = scaler_data.shape[-1]
        mean = [scaler_data[..., i:i + 1][scaler_mask[..., i:i + 1]].mean() for i in range(dim)]
        std = [scaler_data[..., i:i + 1][scaler_mask[..., i:i + 1]].std() for i in range(dim)]
        self.scaler = self.normalizer_class()(mean, std)

        # --- Training split ---
        train_data = self.data[train_range[0]:train_range[1]]
        train_te = self.timestamp[train_range[0]:train_range[1]]
        train_mask = self.mask[train_range[0]:train_range[1]]
        train_mask_eval = self.mask_eval[train_range[0]:train_range[1]]
        train_sample = sliding_window_samples(train_data, sample_len=window_size)
        train_x = train_sample[:, :sample_len, ...][..., :input_dim]
        train_y = train_sample[:, -output_len:, ...][..., :output_dim]
        train_x = self.scaler.transform(train_x)
        train_te = sliding_window_samples(train_te, sample_len=window_size)
        train_ob_mask = sliding_window_samples(train_mask, sample_len=window_size)[..., :input_dim]
        if target_strategy == 'random':
            train_cond_mask = random_mask(train_ob_mask[:, :sample_len], 0, 1).cuda()[..., :input_dim]
        else:
            t_len = train_ob_mask.shape[0]
            train_cond_mask = torch.concat([
                block_mask(
                    train_ob_mask[i, :sample_len].cpu(),
                    target_strategy='hybrid',
                    min_seq=3,
                    max_seq=12,
                ).cuda().unsqueeze(0)
                for i in range(t_len)
            ])[..., :input_dim]
        train_dataset = TrafficDataset(
            history=train_x,
            target=train_y,
            timestamp=train_te,
            cond_mask=train_cond_mask,
            ob_mask=train_ob_mask,
            training=True,
        )

        # --- Validation split ---
        val_data = self.data[val_range[0]:val_range[1]]
        val_te = self.timestamp[val_range[0]:val_range[1]]
        val_mask = self.mask[val_range[0]:val_range[1]]
        val_mask_eval = self.mask_eval[val_range[0]:val_range[1]]
        val_sample = sliding_window_samples(val_data, sample_len=window_size)
        val_x = val_sample[:, :sample_len, ...][..., :input_dim]
        val_y = val_sample[:, -output_len:, ...][..., :output_dim]
        val_x = self.scaler.transform(val_x)
        val_te = sliding_window_samples(val_te, sample_len=window_size)

        val_ob_mask = sliding_window_samples(val_mask_eval, sample_len=window_size)[..., :input_dim]
        val_cond_mask = sliding_window_samples(val_mask, sample_len=window_size)[:, :sample_len][..., :input_dim]
        val_dataset = TrafficDataset(
            history=val_x,
            target=val_y,
            timestamp=val_te,
            cond_mask=val_cond_mask,
            ob_mask=val_ob_mask,
        )

        # --- Test split ---
        test_data = self.data[test_range[0]:test_range[1]]
        test_te = self.timestamp[test_range[0]:test_range[1]]
        test_mask = self.mask[test_range[0]:test_range[1]]
        test_mask_eval = self.mask_eval[test_range[0]:test_range[1]]
        test_sample = sliding_window_samples(test_data, sample_len=window_size)
        test_x = test_sample[:, :sample_len, ...][..., :input_dim]
        test_y = test_sample[:, -output_len:, ...][..., :output_dim]
        test_x = self.scaler.transform(test_x)
        test_te = sliding_window_samples(test_te, sample_len=window_size)

        test_ob_mask = sliding_window_samples(test_mask_eval, sample_len=window_size)[..., :input_dim]
        test_cond_mask = sliding_window_samples(test_mask, sample_len=window_size)[:, :sample_len][..., :input_dim]
        test_dataset = TrafficDataset(
            history=test_x,
            target=test_y,
            timestamp=test_te,
            cond_mask=test_cond_mask,
            ob_mask=test_ob_mask,
        )

        return train_dataset, val_dataset, test_dataset

    def adjacency_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the adjacency matrix and the distance matrix."""
        return self.adj_mx, self.distance_mx

    def normalizer_class(self):
        """Return the normalizer class used for input scaling."""
        return ZScoreNormalizer


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def build_timestamp_features(
    start: str,
    periods: int,
    freq: str,
) -> torch.Tensor:
    """Construct a (T, 5) tensor of temporal features from a date range.

    Features are: month, day-of-month, day-of-week, hour, minute.

    Args:
        start: Start datetime string, e.g. '20180101 00:00:00'.
        periods: Number of time steps to generate.
        freq: Pandas frequency string, e.g. '5min'.

    Returns:
        Integer tensor of shape (periods, 5).
    """
    time = pd.date_range(start=start, periods=periods, freq=freq)

    month = np.reshape(time.month, (-1, 1))
    dayofmonth = np.reshape(time.day, (-1, 1))
    dayofweek = np.reshape(time.weekday, (-1, 1))
    hour = np.reshape(time.hour, (-1, 1))
    minute = np.reshape(time.minute, (-1, 1))

    timestamp = np.concatenate((month, dayofmonth, dayofweek, hour, minute), -1)

    timestamp = torch.tensor(timestamp)

    return timestamp


# Mapping from dataset name to a callable that builds timestamp features
# given the total number of time steps T.
TIMESTAMP_BUILDERS: Dict[str, Callable[[int], torch.Tensor]] = {
    'PEMS08': lambda T: build_timestamp_features(start='20160701 00:00:00', periods=T, freq='5min'),
    'PEMS07': lambda T: build_timestamp_features(start='20170501 00:00:00', periods=T, freq='5min'),
    'PEMS04': lambda T: build_timestamp_features(start='20180101 00:00:00', periods=T, freq='5min'),
    'PEMS03': lambda T: build_timestamp_features(start='20180901 00:00:00', periods=T, freq='5min'),
    'NYCTAXI': lambda T: build_timestamp_features(start='20160401 00:00:00', periods=T, freq='30min'),
    'CHIBIKE': lambda T: build_timestamp_features(start='20160401 00:00:00', periods=T, freq='30min'),
}


# ---------------------------------------------------------------------------
# Concrete reader implementations
# ---------------------------------------------------------------------------

class PEMSFlowReader(TrafficDataLoader):
    """Reader for standard PEMS traffic flow .npz files (PEMS03/04/07/08)."""

    def read_data(
        self,
        data_path: str,
        adj_path: Optional[str] = None,
    ) -> Tuple[
        torch.Tensor, int, int,
        np.ndarray, np.ndarray,
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        data = torch.from_numpy(np.load(data_path)['data'][..., :])

        T, node_num, features = data.shape
        if 'PEMS03' in self.dataset:
            id_filename = adj_path.replace('csv', 'txt')
        else:
            id_filename = None
        adj_mx, distance_mx = load_adjacency_from_csv(adj_path, node_num, id_filename)
        adj_mx = np.where(np.eye(node_num).astype('bool'), 1, adj_mx)

        timestamp = TIMESTAMP_BUILDERS[self.dataset[:6]](T)

        return (
            data, node_num, features,
            adj_mx, distance_mx,
            timestamp, torch.ones_like(data), torch.ones_like(data),
        )


class PEMSMissingReader(TrafficDataLoader):
    """Reader for PEMS datasets with missing-value masks."""

    def read_data(
        self,
        data_path: str,
        adj_path: Optional[str] = None,
    ) -> Tuple[
        torch.Tensor, int, int,
        np.ndarray, np.ndarray,
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        dir_name = os.path.dirname(data_path)
        file_name = os.path.basename(data_path)

        true_datapath = os.path.join(dir_name, file_name.replace('miss', 'true'))
        miss_datapath = os.path.join(dir_name, file_name.replace('true', 'miss'))

        miss_data = np.load(miss_datapath)
        mask = torch.from_numpy(miss_data['mask'][:, :, :].astype('long'))
        data = np.load(true_datapath)['data'].astype(np.float32)[:, :, :]
        data[np.isnan(data)] = 0
        data = torch.from_numpy(data)

        T, node_num, features = data.shape

        adj_mx, distance_mx = load_adjacency_from_csv(adj_path, node_num)
        adj_mx = np.where(np.eye(node_num).astype('bool'), 1, adj_mx)

        timestamp = TIMESTAMP_BUILDERS[self.dataset[:6]](T)

        return (
            data, node_num, features,
            adj_mx, distance_mx,
            timestamp, mask, torch.ones_like(data),
        )


class NYCTaxiReader(TrafficDataLoader):
    """Reader for NYC Taxi / Chicago Bike trip-count datasets."""

    def read_data(
        self,
        data_path: str,
        adj_path: Optional[str] = None,
    ) -> Tuple[
        torch.Tensor, int, int,
        np.ndarray, np.ndarray,
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        data = torch.from_numpy(np.load(data_path)['data'][..., :])
        data = np.transpose(data, (1, 0, 2))

        T, node_num, features = data.shape

        adj_mx = np.ones((node_num, node_num)).astype(np.float32)
        distance_mx = np.ones((node_num, node_num)).astype(np.float32)
        timestamp = TIMESTAMP_BUILDERS[self.dataset](T)

        return (
            data, node_num, features,
            adj_mx, distance_mx,
            timestamp, torch.ones_like(data), torch.ones_like(data),
        )
