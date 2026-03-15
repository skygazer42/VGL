import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, edge_softmax, mean_propagate, sum_propagate


class GENConv(nn.Module):
    def __init__(self, in_channels, out_channels, aggr="softmax", beta=1.0):
        super().__init__()
        self.out_channels = out_channels
        self.aggr = aggr
        self.beta = beta
        self.message_linear = nn.Linear(in_channels, out_channels)
        self.root_linear = nn.Linear(in_channels, out_channels)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "GENConv")
        if self.aggr == "mean":
            return self.message_linear(mean_propagate(x, edge_index)) + self.root_linear(x)
        if self.aggr == "sum":
            return self.message_linear(sum_propagate(x, edge_index)) + self.root_linear(x)
        if self.aggr != "softmax":
            raise ValueError("GENConv only supports softmax, mean, and sum aggregators")

        row, col = edge_index
        messages = self.message_linear(x[row])
        scores = self.beta * messages.mean(dim=-1)
        weights = edge_softmax(scores, edge_index, x.size(0))
        out = torch.zeros(
            x.size(0),
            self.out_channels,
            dtype=x.dtype,
            device=x.device,
        )
        out.index_add_(0, col, messages * weights.unsqueeze(-1))
        return out + self.root_linear(x)
