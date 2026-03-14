import torch
from torch import nn


class MessagePassing(nn.Module):
    def forward(self, graph_or_x, edge_index=None):
        if edge_index is None:
            x = graph_or_x.x
            edge_index = graph_or_x.edge_index
        else:
            x = graph_or_x

        row, col = edge_index
        messages = self.message(x[row], x[col])
        out = torch.zeros_like(x)
        out.index_add_(0, col, messages)
        return self.update(out)

    def message(self, x_j, x_i):
        return x_j

    def update(self, aggr_out):
        return aggr_out
