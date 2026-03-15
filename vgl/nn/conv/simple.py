from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, max_propagate, mean_propagate, sum_propagate


class SimpleConv(nn.Module):
    def __init__(self, aggr="mean"):
        super().__init__()
        self.aggr = aggr

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "SimpleConv")
        if self.aggr == "mean":
            return mean_propagate(x, edge_index)
        if self.aggr == "sum":
            return sum_propagate(x, edge_index)
        if self.aggr == "max":
            return max_propagate(x, edge_index)
        raise ValueError("SimpleConv only supports mean, sum, and max aggregators")
