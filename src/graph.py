import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.data import Data
import torch_geometric_temporal as tgnn
from typing import Sequence,Union

class GraphKernel(T.BaseTransform):
    def __init__(self,bandwith=None):
        self.distances = T.Distance(norm=False)
        self.bandwidth = bandwith
    
    def forward(self, graph: Data)->Data:
        graph = self.distances(graph)
        if self.bandwidth == None:
            self.bandwidth = torch.std(graph.edge_attr)
        graph.edge_attr = torch.exp(-torch.sum(graph.edge_attr**2,dim=-1)/self.bandwidth**2).float()
        return graph

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
            return torch.FloatTensor(self.targets[time_index])
    
    def _get_positions(self,time_index: int):
        if self.positions is None:
            return self.positions
        else:
            if self.positions[time_index] is None:
                return self.positions[time_index]
            else:
                return torch.DoubleTensor(self.positions[time_index])
        
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