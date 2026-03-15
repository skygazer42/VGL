import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, edge_softmax


class AGNNConv(nn.Module):
    def __init__(self, channels, beta=1.0, train_beta=False):
        super().__init__()
        self.channels = channels
        self.out_channels = channels
        if train_beta:
            self.beta = nn.Parameter(torch.tensor(float(beta)))
        else:
            self.register_buffer("beta", torch.tensor(float(beta)))

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "AGNNConv")
        row, col = edge_index
        normalized = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        scores = self.beta * (normalized[row] * normalized[col]).sum(dim=-1)
        weights = edge_softmax(scores, edge_index, x.size(0))
        out = torch.zeros_like(x)
        out.index_add_(0, col, x[row] * weights.unsqueeze(-1))
        return out
