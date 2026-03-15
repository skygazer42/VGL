import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, mean_propagate


class H2GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels * 3, out_channels)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "H2GCNConv")
        hop1 = mean_propagate(x, edge_index)
        hop2 = mean_propagate(hop1, edge_index)
        return self.linear(torch.cat([x, hop1, hop2], dim=-1))
