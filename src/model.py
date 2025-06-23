import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric_temporal.nn.recurrent import GConvLSTM,GConvGRU

from torch import Tensor
from typing import Optional

class STGNN_LSTM(torch.nn.Module):
    def __init__(self, node_features, hidden_features, filter_size):
        super(STGNN_LSTM, self).__init__()
        self.recurrent = GConvLSTM(in_channels=node_features,
                                   out_channels=hidden_features,
                                     K=filter_size)
        self.linear = torch.nn.Linear(in_features=hidden_features, out_features=1)

    def forward(self, x, edge_index, edge_weight,h,c):
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0, c_0

class STGNN_GRU(torch.nn.Module):
    def __init__(self, node_features, hidden_features, filter_size):
        super(STGNN_GRU, self).__init__()
        self.recurrent = GConvGRU(in_channels=node_features,
                                   out_channels=hidden_features,
                                     K=filter_size)
        self.linear = torch.nn.Linear(in_features=hidden_features, out_features=1)

    def forward(self, x, edge_index, edge_weight,h):
        h_0 = self.recurrent(x, edge_index, edge_weight, h)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0

class ClassifierGCN(torch.nn.Module):
    def __init__(self, node_features, hidden_features, num_classes, filter_size):
        super().__init__()
        self.conv = gnn.ChebConv(node_features, hidden_features, filter_size)
        self.lin = nn.Linear(hidden_features, num_classes)
    
    def reset_parameters(self):
        self.conv.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.lin(x)
        return x
