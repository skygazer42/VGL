from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, sum_propagate


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.neigh_linear = nn.Linear(in_channels, out_channels)
        self.root_linear = nn.Linear(in_channels, out_channels)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "GraphConv")
        neigh = sum_propagate(x, edge_index)
        return self.neigh_linear(neigh) + self.root_linear(x)
