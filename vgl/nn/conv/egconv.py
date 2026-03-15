import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, max_propagate, mean_propagate, sum_propagate


class EGConv(nn.Module):
    def __init__(self, in_channels, out_channels, aggregators=("sum", "mean", "max")):
        super().__init__()
        self.aggregators = tuple(aggregators)
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels * len(self.aggregators), out_channels)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "EGConv")
        features = []
        for aggregator in self.aggregators:
            if aggregator == "sum":
                features.append(sum_propagate(x, edge_index))
            elif aggregator == "mean":
                features.append(mean_propagate(x, edge_index))
            elif aggregator == "max":
                features.append(max_propagate(x, edge_index))
            else:
                raise ValueError("EGConv only supports sum, mean, and max aggregators")
        return self.linear(torch.cat(features, dim=-1))
