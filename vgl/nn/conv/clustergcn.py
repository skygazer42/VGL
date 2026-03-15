from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, mean_propagate


class ClusterGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, diag_lambda=0.0):
        super().__init__()
        self.out_channels = out_channels
        self.diag_lambda = diag_lambda
        self.neigh_linear = nn.Linear(in_channels, out_channels)
        self.root_linear = nn.Linear(in_channels, out_channels)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "ClusterGCNConv")
        neigh = mean_propagate(x, edge_index)
        return self.neigh_linear(neigh) + self.root_linear((1 + self.diag_lambda) * x)
