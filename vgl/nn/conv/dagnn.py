import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, propagate_steps


class DAGNNConv(nn.Module):
    def __init__(self, channels, steps=3):
        super().__init__()
        self.channels = channels
        self.out_channels = channels
        self.steps = steps
        self.gate = nn.Linear(channels, 1)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "DAGNNConv")
        outputs = propagate_steps(x, edge_index, self.steps)
        stacked = torch.stack(outputs, dim=1)
        scores = self.gate(stacked).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (stacked * weights).sum(dim=1)
