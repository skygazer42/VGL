import torch

from vgl import Graph
from vgl.ops import edge_subgraph, node_subgraph


def test_node_subgraph_filters_edges_and_relabels_nodes():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 3, 1], [2, 3, 0, 3]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )

    subgraph = node_subgraph(graph, torch.tensor([0, 2, 3]))

    assert torch.equal(subgraph.x, torch.tensor([[1.0], [3.0], [4.0]]))
    assert torch.equal(subgraph.edge_index, torch.tensor([[0, 1, 2], [1, 2, 0]]))


def test_edge_subgraph_filters_edges_and_preserves_node_space():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 3, 1], [2, 3, 0, 3]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        edge_data={"edge_weight": torch.tensor([1.0, 2.0, 3.0, 4.0])},
    )

    subgraph = edge_subgraph(graph, torch.tensor([1, 3]))

    assert torch.equal(subgraph.x, graph.x)
    assert torch.equal(subgraph.edge_index, torch.tensor([[2, 1], [3, 3]]))
    assert torch.equal(subgraph.edata["edge_weight"], torch.tensor([2.0, 4.0]))


def test_hetero_node_subgraph_filters_relation_and_relabels_per_type():
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0], [3.0]])},
            "institution": {"x": torch.tensor([[7.0]])},
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1, 1], [1, 0, 2]]),
                "edge_weight": torch.tensor([1.0, 2.0, 3.0]),
            }
        },
    )

    subgraph = node_subgraph(
        graph,
        {"author": torch.tensor([1]), "paper": torch.tensor([0, 2])},
        edge_type=("author", "writes", "paper"),
    )

    assert set(subgraph.nodes) == {"author", "paper"}
    assert set(subgraph.edges) == {("author", "writes", "paper")}
    assert torch.equal(subgraph.nodes["author"].x, torch.tensor([[20.0]]))
    assert torch.equal(subgraph.nodes["paper"].x, torch.tensor([[1.0], [3.0]]))
    assert torch.equal(subgraph.edges[("author", "writes", "paper")].edge_index, torch.tensor([[0, 0], [0, 1]]))
    assert torch.equal(subgraph.edges[("author", "writes", "paper")].edge_weight, torch.tensor([2.0, 3.0]))


def test_hetero_edge_subgraph_preserves_participating_node_spaces():
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0], [3.0]])},
            "institution": {"x": torch.tensor([[7.0]])},
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1, 1], [1, 0, 2]]),
                "edge_weight": torch.tensor([1.0, 2.0, 3.0]),
            }
        },
    )

    subgraph = edge_subgraph(graph, torch.tensor([0, 2]), edge_type=("author", "writes", "paper"))

    assert set(subgraph.nodes) == {"author", "paper"}
    assert set(subgraph.edges) == {("author", "writes", "paper")}
    assert torch.equal(subgraph.nodes["author"].x, graph.nodes["author"].x)
    assert torch.equal(subgraph.nodes["paper"].x, graph.nodes["paper"].x)
    assert torch.equal(subgraph.edges[("author", "writes", "paper")].edge_index, torch.tensor([[0, 1], [1, 2]]))
    assert torch.equal(subgraph.edges[("author", "writes", "paper")].edge_weight, torch.tensor([1.0, 3.0]))
