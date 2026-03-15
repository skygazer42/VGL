import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs


class ResGatedGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.src_gate_linear = nn.Linear(in_channels, out_channels)
        self.dst_gate_linear = nn.Linear(in_channels, out_channels)
        self.message_linear = nn.Linear(in_channels, out_channels)
        self.root_linear = nn.Linear(in_channels, out_channels)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "ResGatedGraphConv")
        row, col = edge_index
        gate = torch.sigmoid(
            self.src_gate_linear(x[row]) + self.dst_gate_linear(x[col]),
        )
        messages = gate * self.message_linear(x[row])
        out = torch.zeros(
            x.size(0),
            self.out_channels,
            dtype=x.dtype,
            device=x.device,
        )
        out.index_add_(0, col, messages)
        return out + self.root_linear(x)
