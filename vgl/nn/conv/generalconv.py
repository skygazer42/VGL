import torch
import torch.nn.functional as F
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, edge_softmax


class GeneralConv(nn.Module):
    _SUPPORTED_AGGRS = {"add", "mean", "max"}

    def __init__(
        self,
        in_channels,
        out_channels,
        aggr="add",
        skip_linear=False,
        directed_msg=True,
        heads=1,
        attention=False,
        l2_normalize=False,
    ):
        super().__init__()
        if aggr not in self._SUPPORTED_AGGRS:
            raise ValueError(f"GeneralConv received unsupported aggr={aggr!r}")
        if heads < 1:
            raise ValueError("GeneralConv requires heads >= 1")
        if out_channels % heads != 0:
            raise ValueError("GeneralConv requires out_channels to be divisible by heads")

        self.out_channels = out_channels
        self.aggr = aggr
        self.directed_msg = directed_msg
        self.heads = heads
        self.attention = attention
        self.l2_normalize = l2_normalize
        self.head_channels = out_channels // heads

        message_in_channels = in_channels * (2 if directed_msg else 1)
        self.message_linear = nn.Linear(message_in_channels, out_channels)
        self.attention_linear = nn.Linear(message_in_channels, heads, bias=False) if attention else None
        if skip_linear or in_channels != out_channels:
            self.root_linear = nn.Linear(in_channels, out_channels)
        else:
            self.root_linear = nn.Identity()

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "GeneralConv")
        row, col = edge_index
        src = x[row]
        dst = x[col]
        message_inputs = torch.cat([src, dst], dim=-1) if self.directed_msg else src
        messages = self.message_linear(message_inputs).view(-1, self.heads, self.head_channels)

        weights = None
        if self.attention:
            scores = self.attention_linear(message_inputs)
            weights = torch.stack(
                [edge_softmax(scores[:, head], edge_index, x.size(0)) for head in range(self.heads)],
                dim=-1,
            )
            messages = messages * weights.unsqueeze(-1)

        if self.aggr == "max":
            out = torch.full(
                (x.size(0), self.heads, self.head_channels),
                float("-inf"),
                dtype=x.dtype,
                device=x.device,
            )
            index = col.view(-1, 1, 1).expand(-1, self.heads, self.head_channels)
            out.scatter_reduce_(0, index, messages, reduce="amax", include_self=True)
            out = torch.where(torch.isneginf(out), torch.zeros_like(out), out)
        else:
            out = torch.zeros(
                x.size(0),
                self.heads,
                self.head_channels,
                dtype=x.dtype,
                device=x.device,
            )
            out.index_add_(0, col, messages)
            if self.aggr == "mean":
                if weights is None:
                    normalizer = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
                    normalizer.index_add_(0, col, torch.ones(col.size(0), dtype=x.dtype, device=x.device))
                    out = out / normalizer.clamp_min(1).view(-1, 1, 1)
                else:
                    normalizer = torch.zeros(
                        x.size(0),
                        self.heads,
                        dtype=x.dtype,
                        device=x.device,
                    )
                    normalizer.index_add_(0, col, weights)
                    out = out / normalizer.clamp_min(1e-12).unsqueeze(-1)

        out = out.reshape(x.size(0), self.out_channels) + self.root_linear(x)
        if self.l2_normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out
