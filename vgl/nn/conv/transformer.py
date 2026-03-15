import math

import torch
import torch.nn.functional as F
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, edge_softmax


class TransformerConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        concat=True,
        beta=False,
        dropout=0.0,
        bias=True,
        root_weight=True,
    ):
        super().__init__()
        if heads < 1:
            raise ValueError("TransformerConv requires heads >= 1")
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError("TransformerConv requires dropout to be in [0, 1]")

        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.beta = beta and root_weight
        self.dropout = dropout
        self.root_weight = root_weight

        hidden_channels = heads * out_channels
        final_channels = hidden_channels if concat else out_channels
        self.query_linear = nn.Linear(in_channels, hidden_channels, bias=bias)
        self.key_linear = nn.Linear(in_channels, hidden_channels, bias=bias)
        self.value_linear = nn.Linear(in_channels, hidden_channels, bias=bias)
        self.root_linear = nn.Linear(in_channels, final_channels, bias=bias) if root_weight else None
        self.beta_linear = nn.Linear(final_channels * 3, final_channels, bias=bias) if self.beta else None

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "TransformerConv")
        row, col = edge_index
        query = self.query_linear(x).view(x.size(0), self.heads, self.out_channels)
        key = self.key_linear(x).view(x.size(0), self.heads, self.out_channels)
        value = self.value_linear(x).view(x.size(0), self.heads, self.out_channels)

        query_dst = query[col]
        key_src = key[row]
        scores = (query_dst * key_src).sum(dim=-1) / math.sqrt(self.out_channels)
        weights = torch.stack(
            [edge_softmax(scores[:, head], edge_index, x.size(0)) for head in range(self.heads)],
            dim=-1,
        )
        weights = F.dropout(weights, p=self.dropout, training=self.training)

        out = torch.zeros(
            x.size(0),
            self.heads,
            self.out_channels,
            dtype=x.dtype,
            device=x.device,
        )
        out.index_add_(0, col, value[row] * weights.unsqueeze(-1))
        if self.concat:
            out = out.reshape(x.size(0), self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if not self.root_weight:
            return out

        root = self.root_linear(x)
        if not self.beta:
            return out + root

        gate = torch.sigmoid(self.beta_linear(torch.cat([out, root, root - out], dim=-1)))
        return gate * root + (1 - gate) * out
