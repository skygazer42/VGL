import copy
import inspect

import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs


class GroupRevRes(nn.Module):
    def __init__(self, conv, num_groups=None, split_dim=-1):
        super().__init__()
        self.split_dim = split_dim

        if isinstance(conv, nn.ModuleList):
            self.convs = conv
        else:
            if num_groups is None:
                raise ValueError("GroupRevRes requires num_groups when conv is a single module")
            self.convs = nn.ModuleList([conv])
            for _ in range(num_groups - 1):
                cloned = copy.deepcopy(conv)
                if hasattr(cloned, "reset_parameters"):
                    cloned.reset_parameters()
                self.convs.append(cloned)

        if len(self.convs) < 2:
            raise ValueError("GroupRevRes requires at least 2 groups")

        for module in self.convs:
            self._validate_forward_contract(module)

    @property
    def num_groups(self):
        return len(self.convs)

    def reset_parameters(self):
        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "GroupRevRes")
        return self._forward_tensor(x, edge_index)

    def inverse(self, graph_or_x, edge_index=None):
        y, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "GroupRevRes")
        return self._inverse_tensor(y, edge_index)

    def _forward_tensor(self, x, edge_index):
        xs = self._chunk(x)
        ys = []
        running = sum(xs[1:])
        for idx, conv in enumerate(self.convs):
            running = xs[idx] + conv(running, edge_index)
            ys.append(running)
        return torch.cat(ys, dim=self.split_dim)

    def _inverse_tensor(self, y, edge_index):
        ys = self._chunk(y)
        xs = []
        for idx in range(self.num_groups - 1, -1, -1):
            if idx != 0:
                running = ys[idx - 1]
            else:
                running = sum(xs)
            x = ys[idx] - self.convs[idx](running, edge_index)
            xs.append(x)
        return torch.cat(xs[::-1], dim=self.split_dim)

    def _chunk(self, x):
        try:
            channels = x.size(self.split_dim)
        except IndexError as error:
            raise ValueError("GroupRevRes split_dim is out of range for the input tensor") from error

        if channels % self.num_groups != 0:
            raise ValueError("GroupRevRes requires the split dimension to be divisible by num_groups")
        return torch.chunk(x, self.num_groups, dim=self.split_dim)

    def _validate_forward_contract(self, conv):
        signature = inspect.signature(conv.forward)
        required = []
        positional = []
        for parameter in signature.parameters.values():
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                positional.append(parameter)
                if parameter.default is inspect._empty:
                    required.append(parameter)

        if len(required) > 2:
            raise ValueError(
                "GroupRevRes only supports wrapped modules with a homogeneous forward runtime "
                "that does not require extra mandatory arguments"
            )

        if len(positional) < 1:
            raise ValueError("GroupRevRes requires wrapped modules with a callable forward contract")

    def __repr__(self):
        return f"{self.__class__.__name__}({self.convs[0]!r}, num_groups={self.num_groups})"
