import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, propagate_steps


class MixHopConv(nn.Module):
    def __init__(self, in_channels, out_channels, powers=(0, 1, 2)):
        super().__init__()
        self.out_channels = out_channels
        self.powers = tuple(powers)
        self.linear = nn.Linear(in_channels * len(self.powers), out_channels)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "MixHopConv")
        propagated = propagate_steps(x, edge_index, max(self.powers))
        features = [propagated[power] for power in self.powers]
        return self.linear(torch.cat(features, dim=-1))
