import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, propagate_steps


class GPRGNNConv(nn.Module):
    def __init__(self, channels, steps=10, alpha=0.1):
        super().__init__()
        self.channels = channels
        self.out_channels = channels
        self.steps = steps
        self.alpha = alpha
        init = torch.tensor(
            [alpha * (1 - alpha) ** k for k in range(steps)] + [(1 - alpha) ** steps],
            dtype=torch.float32,
        )
        self.gamma = nn.Parameter(init)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "GPRGNNConv")
        outputs = propagate_steps(x, edge_index, self.steps)
        out = torch.zeros_like(x)
        for weight, propagated in zip(self.gamma, outputs):
            out = out + weight * propagated
        return out
