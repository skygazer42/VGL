import torch
from torch import nn


class APPNPConv(nn.Module):
    def __init__(self, in_channels, out_channels, steps=10, alpha=0.1):
        super().__init__()
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, out_channels)
        self.steps = steps
        self.alpha = alpha

    def forward(self, graph_or_x, edge_index=None):
        if edge_index is None:
            if len(graph_or_x.nodes) != 1:
                raise ValueError("APPNPConv currently supports homogeneous graphs only")
            x = graph_or_x.x
            edge_index = graph_or_x.edge_index
        else:
            x = graph_or_x

        row, col = edge_index
        initial = self.linear(x)
        out = initial
        for _ in range(self.steps):
            propagated = torch.zeros_like(out)
            propagated.index_add_(0, col, out[row])
            out = (1 - self.alpha) * propagated + self.alpha * initial
        return out
