import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, mean_propagate


class MFConv(nn.Module):
    def __init__(self, in_channels, out_channels, max_degree=4):
        super().__init__()
        self.out_channels = out_channels
        self.max_degree = max_degree
        self.neigh_linears = nn.ModuleList(
            nn.Linear(in_channels, out_channels) for _ in range(max_degree + 1)
        )
        self.root_linears = nn.ModuleList(
            nn.Linear(in_channels, out_channels) for _ in range(max_degree + 1)
        )

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "MFConv")
        neigh = mean_propagate(x, edge_index)
        _, col = edge_index
        degree = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        degree.index_add_(0, col, torch.ones(col.size(0), dtype=torch.long, device=x.device))
        degree = degree.clamp_max(self.max_degree)

        out = torch.zeros(
            x.size(0),
            self.out_channels,
            dtype=x.dtype,
            device=x.device,
        )
        for bucket in range(self.max_degree + 1):
            mask = degree == bucket
            if mask.any():
                out[mask] = self.neigh_linears[bucket](neigh[mask]) + self.root_linears[bucket](x[mask])
        return out
