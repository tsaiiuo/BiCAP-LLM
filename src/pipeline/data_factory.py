"""Factory functions for constructing complete data pipelines.

Provides ``build_data_pipeline`` which instantiates the appropriate dataset
reader, splits data, creates DataLoaders, and returns everything the trainer
needs in a single call.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.utils.data

from pipeline.loader import PEMSFlowReader, PEMSMissingReader, NYCTaxiReader


# Registry mapping dataset identifiers to their reader classes.
DATASET_REGISTRY = {
    'PEMS08FLOW': PEMSFlowReader,
    'PEMS04FLOW': PEMSFlowReader,
    'PEMS03FLOW': PEMSFlowReader,
    'PEMS07FLOW': PEMSFlowReader,
    'PEMS08MISSING': PEMSMissingReader,
    'PEMS04MISSING': PEMSMissingReader,
    'PEMS03MISSING': PEMSMissingReader,
    'PEMS07MISSING': PEMSMissingReader,
    'NYCTAXI': NYCTaxiReader,
    'CHITAXI': NYCTaxiReader,
}


def create_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
) -> torch.utils.data.DataLoader:
    """Wrap a PyTorch Dataset in a DataLoader with the given settings.

    Args:
        dataset: The dataset to iterate over.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle every epoch.
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        A configured DataLoader instance.
    """
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return dataloader


def build_data_pipeline(
    dataset: str,
    batch_size: int,
    sample_len: int,
    output_len: int,
    window_size: int,
    input_dim: int,
    output_dim: int,
    train_ratio: float,
    val_ratio: float,
    data_path: str,
    adj_path: str,
    target_strategy: str,
    few_shot: float = 1,
    node_shuffle_seed: Optional[int] = None,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    object,
    int,
    int,
    np.ndarray,
    np.ndarray,
]:
    """Build the full data pipeline: read raw files, split, normalise, wrap in loaders.

    Args:
        dataset: Dataset identifier (must be a key in ``DATASET_REGISTRY``).
        batch_size: Batch size for all DataLoaders.
        sample_len: Number of historical time steps.
        output_len: Number of future time steps to predict.
        window_size: Total sliding-window length (input + output).
        input_dim: Number of input feature channels.
        output_dim: Number of output feature channels.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        data_path: Path to the raw data file (.npz).
        adj_path: Path to the adjacency/distance CSV.
        target_strategy: Masking strategy ('random' or 'hybrid').
        few_shot: Fraction of training data to use (default 1 = all).
        node_shuffle_seed: If set, randomly permute node order with this seed.

    Returns:
        A tuple of:
            - train_loader, val_loader, test_loader (DataLoaders)
            - scaler (fitted normalizer)
            - node_num (number of spatial nodes)
            - features (number of raw feature channels)
            - adj_mx (adjacency matrix)
            - distance_mx (distance matrix)
    """
    reader = DATASET_REGISTRY[dataset](data_path, adj_path, dataset, node_shuffle_seed)

    train_set, val_set, test_set = reader.prepare_splits(
        sample_len=sample_len,
        output_len=output_len,
        window_size=window_size,
        input_dim=input_dim,
        output_dim=output_dim,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        target_strategy=target_strategy,
        few_shot=few_shot,
    )

    train_loader = create_dataloader(train_set, batch_size=batch_size)
    val_loader = create_dataloader(val_set, batch_size=batch_size)
    test_loader = create_dataloader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)

    scaler = reader.scaler
    node_num, features = reader.node_num, reader.features

    adj_mx, distance_mx = reader.adjacency_matrix()

    return (
        train_loader, val_loader, test_loader,
        scaler, node_num, features,
        adj_mx, distance_mx,
    )
