import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, propagate_steps


class SSGConv(nn.Module):
    def __init__(self, channels, steps=10, alpha=0.1):
        super().__init__()
        self.channels = channels
        self.out_channels = channels
        self.steps = steps
        self.alpha = alpha

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "SSGConv")
        outputs = propagate_steps(x, edge_index, self.steps)
        if len(outputs) == 1:
            return x
        smoothed = torch.stack(outputs[1:], dim=0).mean(dim=0)
        return self.alpha * x + (1 - self.alpha) * smoothed
