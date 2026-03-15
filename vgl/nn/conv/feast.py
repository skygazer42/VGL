import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, edge_softmax


class FeaStConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4):
        super().__init__()
        self.out_channels = out_channels
        self.heads = heads
        self.value_linear = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.assign_linear = nn.Linear(in_channels, heads)
        self.root_linear = nn.Linear(in_channels, out_channels)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "FeaStConv")
        row, col = edge_index
        values = self.value_linear(x).view(x.size(0), self.heads, self.out_channels)
        logits = self.assign_linear(x[row] - x[col])
        out = torch.zeros(
            x.size(0),
            self.heads,
            self.out_channels,
            dtype=x.dtype,
            device=x.device,
        )
        for head in range(self.heads):
            weights = edge_softmax(logits[:, head], edge_index, x.size(0))
            out[:, head].index_add_(0, col, values[row, head] * weights.unsqueeze(-1))
        return out.mean(dim=1) + self.root_linear(x)
