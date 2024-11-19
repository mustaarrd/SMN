import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# GNN 모델 정의
class GNNModel(nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 2)  # 이진 분류

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.lin(x)
        return x
