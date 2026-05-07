import csv
import numpy as np
import networkx as nx
from pathlib import Path
from typing import List, Optional, Tuple


def compute_laplacian_eigenvectors(
    adj: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute eigenvectors and eigenvalues of the normalized graph Laplacian."""
    num_nodes = adj.shape[0]

    degree_inv_sqrt = np.sum(adj, axis=1)
    degree_inv_sqrt = np.divide(1, np.sqrt(degree_inv_sqrt), out=np.zeros_like(degree_inv_sqrt))
    degree_inv_sqrt = np.nan_to_num(degree_inv_sqrt, nan=0)
    D_inv_sqrt = np.diag(degree_inv_sqrt)

    identity = np.eye(num_nodes)
    laplacian = identity - D_inv_sqrt @ adj @ D_inv_sqrt

    eigvals, eigvecs = np.linalg.eig(laplacian)
    return np.real(eigvecs), np.real(eigvals)  # [N, N], [N]


def normalize_adjacency(W: np.ndarray) -> np.ndarray:
    """Compute row-normalized adjacency with self-loops: D_hat^{-1} A_hat."""
    assert W.shape[0] == W.shape[1]
    num_nodes = W.shape[0]
    A_hat = W + np.identity(num_nodes)
    D_inv = np.diag(1.0 / np.sum(A_hat, axis=1))
    return np.dot(D_inv, A_hat)


def compute_node_ordering(
    adj: np.ndarray,
) -> Tuple[List[int], List[int]]:
    """Topological-style ordering based on greedy minimum in-degree removal."""
    num_nodes = adj.shape[0]
    A_no_self = adj * (1 - np.eye(num_nodes)).astype(np.int32)
    in_degree = adj.sum(axis=0)

    order: List[int] = []
    visited = 0

    while visited != num_nodes:
        node = np.argmin(in_degree)
        order.append(node)
        visited += 1

        outgoing = A_no_self[node, :]
        in_degree -= outgoing
        in_degree[node] = np.inf

    inverse_order = list(np.argsort(order))
    return order, inverse_order


def compute_shortest_paths(
    adj: np.ndarray,
    distance: np.ndarray,
) -> np.ndarray:
    """All-pairs shortest path lengths on a weighted graph; unreachable = inf."""
    num_nodes = adj.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))

    for edge in np.argwhere(adj != 0):
        src, dst = edge
        if src > dst:
            continue
        graph.add_edge(src, dst, weight=distance[src, dst])

    shortest = nx.shortest_path_length(graph, weight='weight')
    result = np.zeros_like(distance)

    for node, path_dict in shortest:
        sorted_items = sorted(path_dict.items(), key=lambda v: v[0])
        distances = [d for _, d in sorted_items]
        result[node] = np.array(distances)

    result = np.where(
        np.logical_or(result != 0, np.eye(N=num_nodes) != 0),
        result,
        np.inf,
    )
    return result


def load_adjacency_from_csv(
    distance_file: str,
    num_nodes: int,
    id_file: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load adjacency and distance matrices from a CSV (or .npy) edge file."""
    filepath = Path(distance_file)

    if filepath.suffix == '.npy':
        adj = np.load(str(filepath))
        return adj, adj.copy()

    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    dist_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    if id_file is not None:
        with open(id_file, 'r') as fid:
            id_map = {
                str(raw_id): index
                for index, raw_id in enumerate(fid.read().strip().split('\n'))
            }

        with open(str(filepath), 'r') as fedge:
            fedge.readline()  # skip header
            reader = csv.reader(fedge)
            for row in reader:
                if len(row) != 3:
                    continue
                src, dst, dist = str(row[0]), str(row[1]), float(row[2])
                adj[id_map[src], id_map[dst]] = 1
                dist_matrix[id_map[src], id_map[dst]] = dist
    else:
        with open(str(filepath), 'r') as fedge:
            fedge.readline()  # skip header
            reader = csv.reader(fedge)
            for row in reader:
                if len(row) != 3:
                    continue
                src, dst, dist = int(row[0]), int(row[1]), float(row[2])
                adj[src, dst] = 1
                dist_matrix[src, dst] = dist

    return adj, dist_matrix


def load_symmetric_adjacency(
    distance_file: str,
    num_nodes: int,
    id_file: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load a symmetric (bidirectional) adjacency matrix from CSV or .npy."""
    filepath = Path(distance_file)

    if filepath.suffix == '.npy':
        adj = np.load(str(filepath))
        return adj, None

    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    dist_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    if id_file is not None:
        with open(id_file, 'r') as fid:
            id_map = {
                str(raw_id): index
                for index, raw_id in enumerate(fid.read().strip().split('\n'))
            }

        with open(str(filepath), 'r') as fedge:
            fedge.readline()  # skip header
            reader = csv.reader(fedge)
            for row in reader:
                if len(row) != 3:
                    continue
                src, dst, dist = str(row[0]), str(row[1]), float(row[2])
                adj[id_map[src], id_map[dst]] = 1
                adj[id_map[dst], id_map[src]] = 1
                dist_matrix[id_map[src], id_map[dst]] = dist
                dist_matrix[id_map[dst], id_map[src]] = dist
    else:
        with open(str(filepath), 'r') as fedge:
            fedge.readline()  # skip header
            reader = csv.reader(fedge)
            for row in reader:
                if len(row) != 3:
                    continue
                src, dst, dist = int(row[0]), int(row[1]), float(row[2])
                adj[src, dst] = 1
                adj[dst, src] = 1
                dist_matrix[src, dst] = dist
                dist_matrix[dst, src] = dist

    return adj, dist_matrix
