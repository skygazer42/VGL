import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, edge_softmax


class FAGCNConv(nn.Module):
    def __init__(self, channels, eps=0.1):
        super().__init__()
        self.channels = channels
        self.out_channels = channels
        self.eps = eps
        self.gate = nn.Linear(channels * 2, 1)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "FAGCNConv")
        row, col = edge_index
        pair = torch.cat([x[row], x[col]], dim=-1)
        scores = torch.tanh(self.gate(pair)).squeeze(-1)
        weights = edge_softmax(scores, edge_index, x.size(0))
        propagated = torch.zeros_like(x)
        propagated.index_add_(0, col, x[row] * weights.unsqueeze(-1))
        return (1 - self.eps) * propagated + self.eps * x
