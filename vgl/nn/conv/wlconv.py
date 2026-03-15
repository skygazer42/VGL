from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, mean_propagate


class WLConvContinuous(nn.Module):
    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "WLConvContinuous")
        return 0.5 * (x + mean_propagate(x, edge_index))
