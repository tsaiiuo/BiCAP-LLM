from typing import Dict, List, Union
from matplotlib import pyplot as plt


def plot_node_mape(mape_per_node: List[float], save_path: str) -> None:
    """Save a line plot of per-node MAPE values."""
    plt.figure()
    plt.plot(range(len(mape_per_node)), mape_per_node, label='mape')
    plt.ylabel('mape')
    plt.xlabel('node')
    plt.savefig(save_path)
    plt.close()


def plot_training_curves(
    train_loss: Dict[str, list],
    val_loss: Dict[str, list],
    save_path: str,
) -> None:
    """Save a line plot comparing training and validation loss over epochs."""
    plt.figure()
    plt.plot(train_loss['x'], train_loss['y'], label='train loss')
    plt.plot(val_loss['x'], val_loss['y'], label='val loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
