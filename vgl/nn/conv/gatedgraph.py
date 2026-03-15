from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, sum_propagate


class GatedGraphConv(nn.Module):
    def __init__(self, channels, steps=3):
        super().__init__()
        self.channels = channels
        self.out_channels = channels
        self.steps = steps
        self.linear = nn.Linear(channels, channels, bias=False)
        self.gru = nn.GRUCell(channels, channels)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "GatedGraphConv")
        if x.size(-1) != self.channels:
            raise ValueError("GatedGraphConv requires input features to match channels")

        state = x
        for _ in range(self.steps):
            propagated = sum_propagate(self.linear(state), edge_index)
            state = self.gru(propagated, state)
        return state
