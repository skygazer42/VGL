import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, symmetric_propagate


class ARMAConv(nn.Module):
    def __init__(self, channels, stacks=1, layers=2, alpha=0.1):
        super().__init__()
        self.channels = channels
        self.out_channels = channels
        self.stacks = stacks
        self.layers = layers
        self.alpha = alpha
        self.stack_weights = nn.Parameter(torch.ones(stacks))

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "ARMAConv")
        stack_outputs = []
        for _ in range(self.stacks):
            current = x
            for _ in range(self.layers):
                propagated = symmetric_propagate(current, edge_index)
                current = (1 - self.alpha) * propagated + self.alpha * x
            stack_outputs.append(current)
        weights = torch.softmax(self.stack_weights, dim=0)
        out = torch.zeros_like(x)
        for weight, stack_out in zip(weights, stack_outputs):
            out = out + weight * stack_out
        return out
