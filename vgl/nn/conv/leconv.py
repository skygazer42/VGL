import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs


class LEConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.src_linear = nn.Linear(in_channels, out_channels)
        self.dst_linear = nn.Linear(in_channels, out_channels)
        self.self_linear = nn.Linear(in_channels, out_channels)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "LEConv")
        row, col = edge_index
        messages = self.src_linear(x[row]) - self.dst_linear(x[col])
        out = torch.zeros(
            x.size(0),
            self.out_channels,
            dtype=x.dtype,
            device=x.device,
        )
        out.index_add_(0, col, messages)
        return self.self_linear(x) + out
