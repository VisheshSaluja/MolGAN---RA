import numpy as np

def mask_graph(adjacency, node_features, mask_ratio=0.1):
    num_nodes = node_features.shape[0]
    mask_indices = np.random.choice(num_nodes, int(num_nodes * mask_ratio), replace=False)
    
    node_features[mask_indices] = 0

    # Handle both 2D (single edge type) and 3D (multiple edge types)
    if adjacency.ndim == 3:
        adjacency[:, mask_indices, :] = 0
        adjacency[:, :, mask_indices] = 0
    elif adjacency.ndim == 2:
        adjacency[mask_indices, :] = 0
        adjacency[:, mask_indices] = 0

    return adjacency, node_features


def deshuffle_graph(adjacency, node_features):
    perm = np.random.permutation(node_features.shape[0])
    
    node_features = node_features[perm]

    if adjacency.ndim == 3:
        adjacency = adjacency[:, perm][:, :, perm]
    elif adjacency.ndim == 2:
        adjacency = adjacency[perm][:, perm]
    
    return adjacency, node_features

def add_noise(node_features, noise_level=0.05):
    """
    Adds Gaussian noise to node features.
    """
    noise = np.random.normal(0, noise_level, node_features.shape)
    return node_features + noise

def apply_corruptions(adjacency, node_features, config):
    """
    Applies selected corruption strategies based on config dictionary.
    Keys:
        - 'mask': True/False
        - 'mask_ratio': Float between 0 and 1
        - 'deshuffle': True/False
        - 'noise': True/False
        - 'noise_level': Float (e.g., 0.05)
    """
    if config.get("mask", False):
        adjacency, node_features = mask_graph(adjacency, node_features, config.get("mask_ratio", 0.1))
    if config.get("deshuffle", False):
        adjacency, node_features = deshuffle_graph(adjacency, node_features)
    if config.get("noise", False):
        node_features = add_noise(node_features, config.get("noise_level", 0.05))
    
    return adjacency, node_features
