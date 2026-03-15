import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs


class FiLMConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.message_linear = nn.Linear(in_channels, out_channels)
        self.film_linear = nn.Linear(in_channels, out_channels * 2)
        self.root_linear = nn.Linear(in_channels, out_channels)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "FiLMConv")
        row, col = edge_index
        messages = self.message_linear(x[row])
        gamma, beta = self.film_linear(x[col]).chunk(2, dim=-1)
        modulated = gamma * messages + beta
        out = torch.zeros(
            x.size(0),
            self.out_channels,
            dtype=x.dtype,
            device=x.device,
        )
        out.index_add_(0, col, modulated)
        return out + self.root_linear(x)
