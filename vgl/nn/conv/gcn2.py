import math

from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, symmetric_propagate


class GCN2Conv(nn.Module):
    def __init__(self, channels, alpha=0.1, theta=1.0, layer=1, shared_weights=True):
        super().__init__()
        self.channels = channels
        self.out_channels = channels
        self.alpha = alpha
        self.theta = theta
        self.layer = layer
        self.shared_weights = shared_weights
        self.linear = nn.Linear(channels, channels)
        self.initial_linear = None if shared_weights else nn.Linear(channels, channels)

    def forward(self, graph_or_x, edge_index=None, x0=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "GCN2Conv")
        if x0 is None:
            raise ValueError("GCN2Conv requires an explicit x0 reference tensor")
        if x.shape != x0.shape:
            raise ValueError("GCN2Conv requires x and x0 to have the same shape")

        propagated = symmetric_propagate(x, edge_index)
        support = (1 - self.alpha) * propagated + self.alpha * x0
        beta = math.log((self.theta / max(self.layer, 1)) + 1.0)
        if self.shared_weights:
            transformed = self.linear(support)
        else:
            transformed = (1 - self.alpha) * self.linear(support) + self.alpha * self.initial_linear(x0)
        return (1 - beta) * support + beta * transformed
