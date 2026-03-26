import pytest
import torch

from vgl import Graph
from vgl.ops import edge_subgraph, in_subgraph, node_subgraph, out_subgraph


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


def test_in_subgraph_filters_homo_edges_and_preserves_node_space():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 3, 1], [2, 3, 0, 3]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        edge_data={"edge_weight": torch.tensor([1.0, 2.0, 3.0, 4.0])},
    )

    subgraph = in_subgraph(graph, torch.tensor([3, 0]))

    assert torch.equal(subgraph.x, graph.x)
    assert torch.equal(subgraph.edge_index, torch.tensor([[2, 3, 1], [3, 0, 3]]))
    assert torch.equal(subgraph.edata["edge_weight"], torch.tensor([2.0, 3.0, 4.0]))
    assert torch.equal(subgraph.edata["e_id"], torch.tensor([1, 2, 3]))


def test_out_subgraph_filters_homo_edges_and_preserves_node_space():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 3, 1], [2, 3, 0, 3]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        edge_data={"edge_weight": torch.tensor([1.0, 2.0, 3.0, 4.0])},
    )

    subgraph = out_subgraph(graph, torch.tensor([0, 3]))

    assert torch.equal(subgraph.x, graph.x)
    assert torch.equal(subgraph.edge_index, torch.tensor([[0, 3], [2, 0]]))
    assert torch.equal(subgraph.edata["edge_weight"], torch.tensor([1.0, 3.0]))
    assert torch.equal(subgraph.edata["e_id"], torch.tensor([0, 2]))


def test_hetero_in_subgraph_filters_all_relations_by_destination_frontier():
    follows = ("user", "follows", "user")
    plays = ("user", "plays", "game")
    graph = Graph.hetero(
        nodes={
            "user": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "game": {"x": torch.tensor([[1.0], [2.0], [3.0]])},
        },
        edges={
            follows: {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
                "weight": torch.tensor([0.1, 0.2, 0.3]),
            },
            plays: {
                "edge_index": torch.tensor([[0, 1, 1], [0, 0, 2]]),
                "hours": torch.tensor([4.0, 5.0, 6.0]),
            },
        },
    )

    subgraph = in_subgraph(graph, {"user": torch.tensor([0, 2]), "game": torch.tensor([2])})

    assert set(subgraph.nodes) == {"user", "game"}
    assert set(subgraph.edges) == {follows, plays}
    assert torch.equal(subgraph.nodes["user"].x, graph.nodes["user"].x)
    assert torch.equal(subgraph.nodes["game"].x, graph.nodes["game"].x)
    assert torch.equal(subgraph.edges[follows].edge_index, torch.tensor([[1, 2], [2, 0]]))
    assert torch.equal(subgraph.edges[follows].weight, torch.tensor([0.2, 0.3]))
    assert torch.equal(subgraph.edges[follows].e_id, torch.tensor([1, 2]))
    assert torch.equal(subgraph.edges[plays].edge_index, torch.tensor([[1], [2]]))
    assert torch.equal(subgraph.edges[plays].hours, torch.tensor([6.0]))
    assert torch.equal(subgraph.edges[plays].e_id, torch.tensor([2]))


def test_hetero_out_subgraph_filters_all_relations_by_source_frontier():
    follows = ("user", "follows", "user")
    plays = ("user", "plays", "game")
    graph = Graph.hetero(
        nodes={
            "user": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
            "game": {"x": torch.tensor([[1.0], [2.0], [3.0]])},
        },
        edges={
            follows: {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
                "weight": torch.tensor([0.1, 0.2, 0.3]),
            },
            plays: {
                "edge_index": torch.tensor([[0, 1, 1], [0, 0, 2]]),
                "hours": torch.tensor([4.0, 5.0, 6.0]),
            },
        },
    )

    subgraph = out_subgraph(graph, {"user": torch.tensor([1])})

    assert set(subgraph.nodes) == {"user", "game"}
    assert set(subgraph.edges) == {follows, plays}
    assert torch.equal(subgraph.edges[follows].edge_index, torch.tensor([[1], [2]]))
    assert torch.equal(subgraph.edges[follows].weight, torch.tensor([0.2]))
    assert torch.equal(subgraph.edges[follows].e_id, torch.tensor([1]))
    assert torch.equal(subgraph.edges[plays].edge_index, torch.tensor([[1, 1], [0, 2]]))
    assert torch.equal(subgraph.edges[plays].hours, torch.tensor([5.0, 6.0]))
    assert torch.equal(subgraph.edges[plays].e_id, torch.tensor([1, 2]))


def test_frontier_subgraph_requires_typed_frontiers_for_multi_type_graphs():
    graph = Graph.hetero(
        nodes={
            "user": {"x": torch.tensor([[1.0]])},
            "game": {"x": torch.tensor([[2.0]])},
        },
        edges={
            ("user", "plays", "game"): {"edge_index": torch.tensor([[0], [0]])},
        },
    )

    with pytest.raises(ValueError, match="keyed by node type"):
        in_subgraph(graph, torch.tensor([0]))
