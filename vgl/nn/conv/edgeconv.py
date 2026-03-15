import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, aggr="max"):
        super().__init__()
        self.out_channels = out_channels
        self.aggr = aggr
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "EdgeConv")
        row, col = edge_index
        messages = self.mlp(torch.cat([x[col], x[row] - x[col]], dim=-1))

        if self.aggr == "sum":
            out = torch.zeros(
                x.size(0),
                self.out_channels,
                dtype=x.dtype,
                device=x.device,
            )
            out.index_add_(0, col, messages)
            return out

        if self.aggr == "mean":
            out = torch.zeros(
                x.size(0),
                self.out_channels,
                dtype=x.dtype,
                device=x.device,
            )
            out.index_add_(0, col, messages)
            degree = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
            degree.index_add_(0, col, torch.ones(col.size(0), dtype=x.dtype, device=x.device))
            return out / degree.clamp_min(1).unsqueeze(-1)

        if self.aggr != "max":
            raise ValueError("EdgeConv only supports max, sum, and mean aggregators")

        out = torch.full(
            (x.size(0), self.out_channels),
            float("-inf"),
            dtype=x.dtype,
            device=x.device,
        )
        index = col.unsqueeze(-1).expand(-1, self.out_channels)
        out.scatter_reduce_(0, index, messages, reduce="amax", include_self=True)
        out = torch.where(torch.isneginf(out), torch.zeros_like(out), out)
        return out
