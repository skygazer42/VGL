from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, sum_propagate, symmetric_propagate


class LGConv(nn.Module):
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "LGConv")
        if self.normalize:
            return symmetric_propagate(x, edge_index)
        return sum_propagate(x, edge_index)
