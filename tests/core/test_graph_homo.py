import torch

from gnn import Graph


def test_homo_graph_exposes_pyg_style_fields():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    x = torch.randn(2, 4)
    y = torch.tensor([0, 1])

    graph = Graph.homo(edge_index=edge_index, x=x, y=y)

    assert torch.equal(graph.edge_index, edge_index)
    assert torch.equal(graph.x, x)
    assert torch.equal(graph.y, y)
    assert torch.equal(graph.ndata["x"], x)
