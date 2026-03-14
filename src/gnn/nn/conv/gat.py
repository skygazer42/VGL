from torch import nn

from gnn.nn.message_passing import MessagePassing


class GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def update(self, aggr_out):
        return self.linear(aggr_out)
