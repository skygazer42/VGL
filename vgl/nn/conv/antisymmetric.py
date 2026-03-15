import torch
import torch.nn.functional as F
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, symmetric_propagate


class AntiSymmetricConv(nn.Module):
    _ACTIVATIONS = {
        "relu": F.relu,
        "tanh": torch.tanh,
    }

    def __init__(self, channels, num_iters=1, epsilon=0.1, gamma=0.1, act="tanh"):
        super().__init__()
        if act not in self._ACTIVATIONS:
            raise ValueError(f"AntiSymmetricConv received unsupported act={act!r}")

        self.channels = channels
        self.out_channels = channels
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.gamma = gamma
        self.activation = self._ACTIVATIONS[act]
        self.weight = nn.Parameter(torch.empty(channels, channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "AntiSymmetricConv")
        state = x
        eye = torch.eye(self.channels, dtype=x.dtype, device=x.device)
        operator = self.weight - self.weight.transpose(0, 1) - (self.gamma * eye)
        for _ in range(self.num_iters):
            propagated = symmetric_propagate(state, edge_index)
            update = self.activation(state @ operator + propagated + self.bias)
            state = state + (self.epsilon * update)
        return state
