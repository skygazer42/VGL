import torch
from torch import nn


class GATv2Conv(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=True):
        super().__init__()
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.linear = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.att = nn.Parameter(torch.randn(heads, out_channels))

    def forward(self, graph_or_x, edge_index=None):
        if edge_index is None:
            if len(graph_or_x.nodes) != 1:
                raise ValueError("GATv2Conv currently supports homogeneous graphs only")
            x = graph_or_x.x
            edge_index = graph_or_x.edge_index
        else:
            x = graph_or_x

        row, col = edge_index
        projected = self.linear(x).view(x.size(0), self.heads, self.out_channels)
        src = projected[row]
        dst = projected[col]
        scores = torch.tanh(src + dst)
        scores = (scores * self.att).sum(dim=-1)
        scores = torch.exp(scores - scores.amax(dim=0, keepdim=True))

        out = torch.zeros(
            x.size(0),
            self.heads,
            self.out_channels,
            dtype=projected.dtype,
            device=projected.device,
        )
        normalizer = torch.zeros(
            x.size(0),
            self.heads,
            dtype=projected.dtype,
            device=projected.device,
        )
        out.index_add_(0, col, src * scores.unsqueeze(-1))
        normalizer.index_add_(0, col, scores)
        out = out / normalizer.clamp_min(1e-12).unsqueeze(-1)
        if self.concat:
            return out.reshape(x.size(0), self.heads * self.out_channels)
        return out.mean(dim=1)
