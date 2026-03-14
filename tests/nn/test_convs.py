import torch

from gnn import Graph
from gnn.nn.conv.gcn import GCNConv
from gnn.nn.conv.sage import SAGEConv


def test_gcn_conv_accepts_graph_input():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
    )
    conv = GCNConv(in_channels=4, out_channels=3)

    out = conv(graph)

    assert out.shape == (2, 3)


def test_sage_conv_accepts_x_and_edge_index():
    x = torch.randn(2, 4)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    conv = SAGEConv(in_channels=4, out_channels=3)

    out = conv(x, edge_index)

    assert out.shape == (2, 3)
