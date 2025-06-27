import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

class ClassifierGCN(torch.nn.Module):
    def __init__(self, node_features, hidden_features, num_classes):
        super().__init__()
        self.conv = gnn.GCNConv(node_features, hidden_features)
        self.lin = nn.Linear(hidden_features, num_classes)
    
    def reset_parameters(self):
        self.conv.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.lin(x)
        return x
