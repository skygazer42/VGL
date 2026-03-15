import torch
from torch import nn

from vgl.nn.conv._homo import (
    coerce_homo_inputs,
    degree_reference_from_histogram,
    max_propagate,
    mean_propagate,
    node_degree,
    sum_propagate,
)


class PNAConv(nn.Module):
    _SUPPORTED_AGGREGATORS = {
        "sum": sum_propagate,
        "mean": mean_propagate,
        "max": max_propagate,
    }
    _SUPPORTED_SCALERS = {"identity", "amplification", "attenuation"}

    def __init__(
        self,
        in_channels,
        out_channels,
        aggregators=("sum", "mean", "max"),
        scalers=("identity", "amplification", "attenuation"),
        deg=None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.aggregators = tuple(aggregators)
        self.scalers = tuple(scalers)
        unsupported_aggr = sorted(set(self.aggregators) - set(self._SUPPORTED_AGGREGATORS))
        if unsupported_aggr:
            raise ValueError(f"PNAConv received unsupported aggregator(s): {unsupported_aggr}")
        unsupported_scalers = sorted(set(self.scalers) - self._SUPPORTED_SCALERS)
        if unsupported_scalers:
            raise ValueError(f"PNAConv received unsupported scaler(s): {unsupported_scalers}")

        self.message_linear = nn.Linear(
            in_channels * len(self.aggregators) * len(self.scalers),
            out_channels,
        )
        self.root_linear = nn.Linear(in_channels, out_channels)
        histogram = torch.as_tensor([], dtype=torch.float32) if deg is None else torch.as_tensor(deg, dtype=torch.float32)
        self.register_buffer("deg_histogram", histogram, persistent=False)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "PNAConv")
        degree = node_degree(edge_index, x.size(0), x.dtype, x.device)
        if self.deg_histogram.numel() > 0:
            degree_reference = degree_reference_from_histogram(self.deg_histogram, x.dtype, x.device)
        else:
            degree_reference = torch.log1p(degree + 1.0).mean().clamp_min(1.0)

        scaled_features = []
        degree_term = torch.log1p(degree + 1.0).unsqueeze(-1)
        for aggregator in self.aggregators:
            aggregated = self._SUPPORTED_AGGREGATORS[aggregator](x, edge_index)
            for scaler in self.scalers:
                if scaler == "identity":
                    scaled_features.append(aggregated)
                elif scaler == "amplification":
                    scaled_features.append(aggregated * (degree_term / degree_reference))
                else:
                    scaled_features.append(aggregated * (degree_reference / degree_term.clamp_min(1e-6)))

        stacked = torch.cat(scaled_features, dim=-1)
        return self.message_linear(stacked) + self.root_linear(x)
