from typing import Optional
import torch
from torch import Tensor
import torch_geometric.transforms as T
from torch_geometric.data import Data
import esig
import numpy as np
from ssegnn.graph import StaticGraphTemporalSignal, GeometricGraph

class SignatureFeatures(T.BaseTransform):
    def __init__(self, sig_depth=3, normalize=True, log_signature=False, time_augment=False, lead_lag=False):
        super().__init__()
        self.sig_depth = sig_depth
        self.normalize = normalize
        self.time_augment = time_augment
        self.lead_lag = lead_lag
        self.log_signature = log_signature

    def lead_lag_transform(self, x_seq):
        batch_size, seq_length, feature_dim = x_seq.size()
        # Initialize an empty tensor for the lead-lag path
        lead_lag_seq = torch.zeros(batch_size, 2 * seq_length - 1, feature_dim * 2, device=x_seq.device, dtype=x_seq.dtype)
        # Even indices (lag equals lead)
        lead_lag_seq[:, 0::2, :] = torch.cat((x_seq, x_seq), dim=-1)
        # Odd indices (lead is one step ahead)
        lead_lag_seq[:, 1::2, :] = torch.cat((x_seq[:, 1:, :], x_seq[:, :-1, :]), dim=-1)
        return lead_lag_seq
    
    def time_augment_transform(self, x_seq):
        batch_size, seq_length, feature_dim = x_seq.size()
        # Create a time vector normalized between 0 and 1
        time_vector = torch.linspace(0, 1, steps=seq_length, device=x_seq.device, dtype=x_seq.dtype)
        # Repeat and reshape to match x_seq dimensions
        time_feature = time_vector.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1)
        # Concatenate along the feature dimension
        x_aug = torch.cat([x_seq, time_feature], dim=-1)
        return x_aug

    def forward(self, dataset: StaticGraphTemporalSignal) -> Data:
        y = dataset[-1].y
        pos = dataset[-1].pos
        # Initialize the feature sequence
        x_seq = torch.zeros([dataset.num_nodes, dataset.snapshot_count, dataset.num_node_features])
        for time, feature in enumerate(dataset.features):
            # Check if `feature` is a numpy array, and if so, convert it
            if isinstance(feature, np.ndarray):
                feature = torch.from_numpy(feature).float()
            x_seq[:, time, :] = feature
        # Apply lead-lag if enabled
        if self.lead_lag:
            x_seq = self.lead_lag_transform(x_seq)
        # Apply time augmentation if enabled
        if self.time_augment:
            x_seq = self.time_augment_transform(x_seq)
        # esig expects numpy arrays of shape (path_length, channels)
        # Compute signature/logsignature for each node
        x_seq_np = x_seq.cpu().numpy()  # shape: (num_nodes, seq_len, features)
        sigs = []
        for node_path in x_seq_np:
            if self.log_signature:
                sig = esig.stream2logsig(node_path, self.sig_depth)
            else:
                sig = esig.stream2sig(node_path, self.sig_depth)
            sigs.append(sig)
        x = torch.tensor(np.stack(sigs), dtype=torch.float32)
        # Normalize if required
        if self.normalize:
            std_x = torch.std(x, dim=0)
            mean_x = torch.mean(x, dim=0)
            x = (x - mean_x) / (std_x + 1e-8)  # Add small epsilon to avoid division by zero
        # Create the static graph dataset with transformed features
        dataset_static = GeometricGraph(x=x, y=y, edge_index=dataset.edge_index, edge_weight=dataset.edge_weight, pos=pos)
        return dataset_static


class RandomFeatures(T.BaseTransform):
    def __init__(self, feature_dim=10, normalize=True, seed=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.normalize = normalize
        self.seed = seed

    def forward(self, dataset: StaticGraphTemporalSignal) -> Data:
        y = dataset[-1].y
        pos = dataset[-1].pos
        # Set random seed for reproducibility if provided
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        # Generate random uncorrelated features for each node
        # Using normal distribution to ensure features are uncorrelated
        x = torch.randn(dataset.num_nodes, self.feature_dim)
        # Normalize if required
        if self.normalize:
            std_x = torch.std(x, dim=0)
            mean_x = torch.mean(x, dim=0)
            x = (x - mean_x) / (std_x + 1e-8)  # Add small epsilon to avoid division by zero
        # Create the static graph dataset with random features
        dataset_static = GeometricGraph(x=x, y=y, edge_index=dataset.edge_index, edge_weight=dataset.edge_weight, pos=pos)
        return dataset_static