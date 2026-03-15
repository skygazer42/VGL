import math

import torch
import torch.nn.functional as F
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, edge_softmax


def _add_self_loops(edge_index, num_nodes):
    row, col = edge_index
    mask = row != col
    filtered = edge_index[:, mask]
    loops = torch.arange(num_nodes, device=edge_index.device, dtype=edge_index.dtype)
    self_loops = torch.stack([loops, loops], dim=0)
    return torch.cat([filtered, self_loops], dim=1)


class SuperGATConv(nn.Module):
    _SUPPORTED_ATTENTION_TYPES = {"MX", "SD"}

    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        concat=True,
        negative_slope=0.2,
        dropout=0.0,
        add_self_loops=True,
        bias=True,
        attention_type="MX",
    ):
        super().__init__()
        if heads < 1:
            raise ValueError("SuperGATConv requires heads >= 1")
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError("SuperGATConv requires dropout to be in [0, 1]")
        if attention_type not in self._SUPPORTED_ATTENTION_TYPES:
            raise ValueError("SuperGATConv only supports attention_type='MX' or 'SD'")

        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.attention_type = attention_type

        hidden_channels = heads * out_channels
        self.linear = nn.Linear(in_channels, hidden_channels, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(heads, out_channels))
        final_channels = hidden_channels if concat else out_channels
        self.bias = nn.Parameter(torch.zeros(final_channels)) if bias else None
        self.register_buffer("_attention_loss", torch.tensor(0.0), persistent=False)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "SuperGATConv")
        if self.add_self_loops:
            edge_index = _add_self_loops(edge_index, x.size(0))

        row, col = edge_index
        projected = self.linear(x).view(x.size(0), self.heads, self.out_channels)
        src = projected[row]
        dst = projected[col]

        if self.attention_type == "MX":
            logits = (
                (src * self.att_src.unsqueeze(0)).sum(dim=-1)
                + (dst * self.att_dst.unsqueeze(0)).sum(dim=-1)
            )
            logits = F.leaky_relu(logits, negative_slope=self.negative_slope)
        else:
            logits = (src * dst).sum(dim=-1) / math.sqrt(self.out_channels)

        weights = torch.stack(
            [edge_softmax(logits[:, head], edge_index, x.size(0)) for head in range(self.heads)],
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
        out.index_add_(0, col, src * weights.unsqueeze(-1))
        if self.concat:
            out = out.reshape(x.size(0), self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        if self.bias is not None:
            out = out + self.bias

        self._attention_loss = self._compute_attention_loss(projected, edge_index, logits)
        return out

    def get_attention_loss(self):
        return self._attention_loss

    def _compute_attention_loss(self, projected, edge_index, pos_logits):
        row, col = edge_index
        positive_scores = pos_logits.mean(dim=-1)

        neg_row = torch.randint(0, projected.size(0), (row.size(0),), device=row.device)
        neg_col = torch.randint(0, projected.size(0), (col.size(0),), device=col.device)
        neg_src = projected[neg_row]
        neg_dst = projected[neg_col]
        if self.attention_type == "MX":
            negative_scores = (
                (neg_src * self.att_src.unsqueeze(0)).sum(dim=-1)
                + (neg_dst * self.att_dst.unsqueeze(0)).sum(dim=-1)
            )
            negative_scores = F.leaky_relu(negative_scores, negative_slope=self.negative_slope).mean(dim=-1)
        else:
            negative_scores = ((neg_src * neg_dst).sum(dim=-1) / math.sqrt(self.out_channels)).mean(dim=-1)

        logits = torch.cat([positive_scores, negative_scores], dim=0)
        labels = torch.cat(
            [
                torch.ones_like(positive_scores),
                torch.zeros_like(negative_scores),
            ],
            dim=0,
        )
        return F.binary_cross_entropy_with_logits(logits, labels)
