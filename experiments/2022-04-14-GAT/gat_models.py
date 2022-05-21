import torch.nn as nn
from torch_geometric.nn import GATConv
from typing import Tuple


class GatClassificationV1(nn.Module):
    def __init__(
        self,
        node_feat: int,  # dimensions of node feature vector
        edge_dim: int,  # dimensions of edge feature vector
        heads: Tuple[int, int, int],  # self attention heads for layers 1, 2, 3
        dropout1: float = 0.5,  # dropout used inside GATConv
        dropout2: float = 0.5,  # dropout used between conv layers
        num_classes: int = 2,  # normal, abnormal growth
    ):
        super().__init__()

        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.dropout = nn.Dropout(p=self.dropout2)
        self.elu = nn.ELU()
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.out_layer_1 = 8
        self.out_layer_2 = 8

        self.conv1 = GATConv(
            in_channels=node_feat,
            out_channels=self.out_layer_1,
            concat=True,
            heads=heads[0],
            dropout=self.dropout1,
            edge_dim=edge_dim,
        )

        self.conv2 = GATConv(
            in_channels=self.out_layer_1 * heads[0],
            out_channels=self.out_layer_2,
            concat=False,
            heads=heads[1],
            dropout=self.dropout1,
            edge_dim=None,
        )

        self.conv3 = GATConv(
            in_channels=self.out_layer_2,
            out_channels=num_classes,
            concat=False,
            heads=heads[2],
            dropout=dropout1,
            edge_dim=None,
        )

    def forward(self, data):
        x = self.dropout(data.x)
        x = self.conv1(data.x, data.edge_index, edge_attr=data.edge_attr)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, data.edge_index)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.conv3(x, data.edge_index)
        x = self.log_softmax(x)

        return x
