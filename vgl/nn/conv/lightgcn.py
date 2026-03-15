from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, symmetric_propagate


class LightGCNConv(nn.Module):
    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "LightGCNConv")
        return symmetric_propagate(x, edge_index)
