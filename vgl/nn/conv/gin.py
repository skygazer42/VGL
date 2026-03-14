import torch
from torch import nn


class GINConv(nn.Module):
    def __init__(self, in_channels, out_channels, eps=0.0, train_eps=False):
        super().__init__()
        self.out_channels = out_channels
        if train_eps:
            self.eps = nn.Parameter(torch.tensor(float(eps)))
        else:
            self.register_buffer("eps", torch.tensor(float(eps)))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, graph_or_x, edge_index=None):
        if edge_index is None:
            if len(graph_or_x.nodes) != 1:
                raise ValueError("GINConv currently supports homogeneous graphs only")
            x = graph_or_x.x
            edge_index = graph_or_x.edge_index
        else:
            x = graph_or_x

        row, col = edge_index
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, col, x[row])
        return self.mlp((1 + self.eps) * x + aggregated)
