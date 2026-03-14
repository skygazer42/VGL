import torch
from torch import nn

from gnn.nn.message_passing import MessagePassing


class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels * 2, out_channels)

    def forward(self, graph_or_x, edge_index=None):
        if edge_index is None:
            x = graph_or_x.x
            edge_index = graph_or_x.edge_index
        else:
            x = graph_or_x

        neigh = super().forward(x, edge_index)
        return self.linear(torch.cat([x, neigh], dim=-1))
