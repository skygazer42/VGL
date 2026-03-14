from torch import nn


class HeteroConv(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.modules_by_type = nn.ModuleDict(modules)

    def forward(self, inputs):
        return {
            edge_type: module(graph_view)
            for edge_type, (module, graph_view) in inputs.items()
        }
