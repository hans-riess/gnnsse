import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.data import Data
import torch_geometric_temporal as tgnn
from typing import Union

class GeometricGraph(Data):
    def __init__(self, edge_index, x, y, edge_weight,pos):
        super().__init__(edge_index=edge_index, edge_attr=edge_weight,x=x,y=y,pos=pos)
        

class StaticGraphTemporalSignal(tgnn.signal.StaticGraphTemporalSignal):
    def __init__(self,
                edge_index,
                edge_weight,
                features,
                targets,
                positions = None,
                **kwargs
    ):
        super().__init__(edge_index, edge_weight, features, targets)
        self.positions = positions
        self.num_nodes = self.features[0].shape[0]
        self.num_node_features = self.features[0].shape[-1]
        self.y = self._get_target(-1)
        self.graph = GeometricGraph(edge_index=edge_index,edge_weight=edge_weight,pos=positions,x=None,y=None)
    
    def _get_target(self, time_index: int):
        if self.targets[time_index] is None:
            return self.targets[time_index]
        else:
            target = self.targets[time_index]
            if isinstance(target, torch.Tensor):
                return target.double()
            else:
                return torch.tensor(target, dtype=torch.double)
    
    def _get_positions(self,time_index: int):
        if self.positions is None:
            return self.positions
        else:
            if self.positions[time_index] is None:
                return self.positions[time_index]
            else:
                pos = self.positions[time_index]
                if isinstance(pos, torch.Tensor):
                    return pos.float()
                else:
                    return torch.tensor(pos, dtype=torch.float)
        
    def __getitem__(self, time_index: Union[int, slice]):
        if isinstance(time_index, slice):
            snapshot = StaticGraphTemporalSignal(
                self.edge_index,
                self.edge_weight,
                self.features[time_index],
                self.targets[time_index],
                self.positions[time_index],
                **{key: getattr(self, key)[time_index] for key in self.additional_feature_keys}
            )
        else:
            x = self._get_features(time_index)
            edge_index = self._get_edge_index()
            edge_weight = self._get_edge_weight()
            y = self._get_target(time_index)
            pos = self._get_positions(time_index)
            additional_features = self._get_additional_features(time_index)

            snapshot = Data(x=x, edge_index=edge_index, edge_attr=edge_weight,
                            y=y, pos=pos, **additional_features)
        return snapshot

    def _get_feature_matrix(self,numpy=False):
        X = torch.zeros([self.num_nodes,self.snapshot_count,self.num_node_features])
        y = torch.zeros([self.snapshot_count])
        for time in range(self.snapshot_count):
            X[:,time,:] = torch.tensor(self.features[time])
            y[time] = torch.tensor(self.targets[time])
        if numpy:
            return X.numpy(),y.numpy()
        else:
            return X,y

    def _get_features(self, time_index: int):
        """Override parent method to handle both numpy arrays and tensors."""
        if isinstance(self.features[time_index], torch.Tensor):
            # If it's already a tensor, just return it as float
            return self.features[time_index].float()
        else:
            # If it's a numpy array, convert to float tensor
            return torch.tensor(self.features[time_index], dtype=torch.float)

class NodeSplitMask(T.BaseTransform):
    def __init__(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=None):
        super().__init__()
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def forward(self, data):
        num_nodes = data.x.size(0)
        # Exclude nodes with label -1 from splitting
        valid_mask = (data.y != -1)
        valid_indices = valid_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        rng = np.random.default_rng(self.seed)
        rng.shuffle(valid_indices)

        num_valid = len(valid_indices)
        train_end = int(self.train_ratio * num_valid)
        val_end = train_end + int(self.val_ratio * num_valid)

        train_idx = valid_indices[:train_end]
        val_idx = valid_indices[train_end:val_end]
        test_idx = valid_indices[val_end:]

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        # Ensure nodes with label -1 are not in any split
        train_mask[data.y == -1] = False
        val_mask[data.y == -1] = False
        test_mask[data.y == -1] = False

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        return data

class GraphNormalize(T.BaseTransform):
    """
    A transform that normalizes node features and handles NaN values.
    
    Args:
        normalize (bool): Whether to normalize features (zero mean, unit variance)
        fill_nan (str): Method to fill NaN values ('forward', 'backward', 'both', or None)
        feature_dim (int): Dimension to normalize over (default: -1 for last dimension)
        eps (float): Small value to avoid division by zero in normalization
    """
    def __init__(self, normalize=True, fill_nan='both', feature_dim=-1, eps=1e-8):
        super().__init__()
        self.normalize = normalize
        self.fill_nan = fill_nan
        self.feature_dim = feature_dim
        self.eps = eps
        
    def forward(self, data):
        if hasattr(data, 'x') and data.x is not None:
            x = data.x.clone()
            
            # Handle NaN values first
            if self.fill_nan and torch.isnan(x).any():
                x = self._fill_nans(x)
            
            # Normalize if requested
            if self.normalize:
                x = self._normalize_features(x)
            
            data.x = x
            
        return data
    
    def _fill_nans(self, x):
        """Fill NaN values using forward/backward fill."""
        if self.fill_nan == 'forward':
            # Forward fill along feature dimension
            x = self._forward_fill(x)
        elif self.fill_nan == 'backward':
            # Backward fill along feature dimension
            x = self._backward_fill(x)
        elif self.fill_nan == 'both':
            # Forward fill then backward fill
            x = self._forward_fill(x)
            x = self._backward_fill(x)
        
        return x
    
    def _forward_fill(self, x):
        """Forward fill NaN values along feature dimension."""
        # Create a mask for NaN values
        nan_mask = torch.isnan(x)
        
        # Forward fill: replace NaN with previous non-NaN value
        for i in range(1, x.shape[0]):
            # For each node, if current value is NaN, use previous node's value
            x[i] = torch.where(nan_mask[i], x[i-1], x[i])
        
        return x
    
    def _backward_fill(self, x):
        """Backward fill NaN values along feature dimension."""
        # Create a mask for NaN values
        nan_mask = torch.isnan(x)
        
        # Backward fill: replace NaN with next non-NaN value
        for i in range(x.shape[0]-2, -1, -1):
            # For each node, if current value is NaN, use next node's value
            x[i] = torch.where(nan_mask[i], x[i+1], x[i])
        
        return x
    
    def _normalize_features(self, x):
        """Normalize features to zero mean and unit variance."""
        # Calculate mean and std along the specified dimension
        if self.feature_dim == -1:
            # Normalize each feature independently
            mean = torch.nanmean(x, dim=0, keepdim=True)
            # Custom nanstd implementation
            std = torch.sqrt(torch.nanmean((x - mean) ** 2, dim=0, keepdim=True))
        else:
            # Normalize along the specified dimension
            mean = torch.nanmean(x, dim=self.feature_dim, keepdim=True)
            # Custom nanstd implementation
            std = torch.sqrt(torch.nanmean((x - mean) ** 2, dim=self.feature_dim, keepdim=True))
        
        # Avoid division by zero
        std = torch.clamp(std, min=self.eps)
        
        # Normalize
        x_norm = (x - mean) / std
        
        # Replace any remaining NaN values with 0
        x_norm = torch.nan_to_num(x_norm, nan=0.0)
        
        return x_norm

class TemporalGraphNormalize(T.BaseTransform):
    """
    A transform that normalizes temporal graph features and handles NaN values.
    Works with StaticGraphTemporalSignal objects.
    
    Args:
        normalize (bool): Whether to normalize features
        fill_nan (str): Method to fill NaN values ('forward', 'backward', 'both', or None)
        feature_dim (int): Dimension to normalize over
        eps (float): Small value to avoid division by zero
    """
    def __init__(self, normalize=True, fill_nan='both', feature_dim=-1, eps=1e-8):
        super().__init__()
        self.normalize = normalize
        self.fill_nan = fill_nan
        self.feature_dim = feature_dim
        self.eps = eps
        
    def forward(self, data):
        if hasattr(data, 'features') and data.features is not None:
            # Process each temporal snapshot
            for i in range(len(data.features)):
                if isinstance(data.features[i], torch.Tensor):
                    x = data.features[i].clone()
                else:
                    x = torch.tensor(data.features[i])
                
                # Handle NaN values
                if self.fill_nan and torch.isnan(x).any():
                    x = self._fill_nans(x)
                
                # Normalize if requested
                if self.normalize:
                    x = self._normalize_features(x)
                
                data.features[i] = x
                
        return data
    
    def _fill_nans(self, x):
        """Fill NaN values using forward/backward fill."""
        if self.fill_nan == 'forward':
            x = self._forward_fill(x)
        elif self.fill_nan == 'backward':
            x = self._backward_fill(x)
        elif self.fill_nan == 'both':
            x = self._forward_fill(x)
            x = self._backward_fill(x)
        return x
    
    def _forward_fill(self, x):
        """Forward fill NaN values along node dimension."""
        nan_mask = torch.isnan(x)
        for i in range(1, x.shape[0]):
            x[i] = torch.where(nan_mask[i], x[i-1], x[i])
        return x
    
    def _backward_fill(self, x):
        """Backward fill NaN values along node dimension."""
        nan_mask = torch.isnan(x)
        for i in range(x.shape[0]-2, -1, -1):
            x[i] = torch.where(nan_mask[i], x[i+1], x[i])
        return x
    
    def _normalize_features(self, x):
        """Normalize features to zero mean and unit variance."""
        if self.feature_dim == -1:
            mean = torch.nanmean(x, dim=0, keepdim=True)
            # Custom nanstd implementation
            std = torch.sqrt(torch.nanmean((x - mean) ** 2, dim=0, keepdim=True))
        else:
            mean = torch.nanmean(x, dim=self.feature_dim, keepdim=True)
            # Custom nanstd implementation
            std = torch.sqrt(torch.nanmean((x - mean) ** 2, dim=self.feature_dim, keepdim=True))
        
        std = torch.clamp(std, min=self.eps)
        x_norm = (x - mean) / std
        x_norm = torch.nan_to_num(x_norm, nan=0.0)
        return x_norm