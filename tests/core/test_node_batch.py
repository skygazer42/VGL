import torch

from vgl import Graph
from vgl.core.batch import NodeBatch
from vgl.data.sample import SampleRecord


def _sample(graph, seed, sample_id):
    return SampleRecord(
        graph=graph,
        metadata={"seed": seed, "sample_id": sample_id},
        sample_id=sample_id,
        subgraph_seed=seed,
    )


def test_node_batch_batches_subgraphs_into_disjoint_union():
    g1 = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
    )
    g2 = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 4),
        y=torch.tensor([1, 0, 1]),
    )

    batch = NodeBatch.from_samples([_sample(g1, 1, "a"), _sample(g2, 2, "b")])

    assert batch.graph.x.size(0) == 5
    assert torch.equal(batch.graph.edge_index, torch.tensor([[0, 2, 3], [1, 3, 4]]))
    assert torch.equal(batch.seed_index, torch.tensor([1, 4]))
    assert batch.metadata == [{"seed": 1, "sample_id": "a"}, {"seed": 2, "sample_id": "b"}]


def test_node_batch_batches_hetero_subgraphs_with_seed_type_offsets():
    g1 = Graph.hetero(
        nodes={
            "paper": {"x": torch.randn(2, 4), "y": torch.tensor([0, 1])},
            "author": {"x": torch.randn(1, 4)},
        },
        edges={
            ("author", "writes", "paper"): {"edge_index": torch.tensor([[0], [1]])},
            ("paper", "written_by", "author"): {"edge_index": torch.tensor([[1], [0]])},
        },
    )
    g2 = Graph.hetero(
        nodes={
            "paper": {"x": torch.randn(1, 4), "y": torch.tensor([1])},
            "author": {"x": torch.randn(2, 4)},
        },
        edges={
            ("author", "writes", "paper"): {"edge_index": torch.tensor([[1], [0]])},
            ("paper", "written_by", "author"): {"edge_index": torch.tensor([[0], [1]])},
        },
    )

    batch = NodeBatch.from_samples(
        [
            SampleRecord(graph=g1, metadata={"seed": 1, "node_type": "paper"}, sample_id="a", subgraph_seed=1),
            SampleRecord(graph=g2, metadata={"seed": 0, "node_type": "paper"}, sample_id="b", subgraph_seed=0),
        ]
    )

    assert batch.graph.nodes["paper"].x.size(0) == 3
    assert batch.graph.nodes["author"].x.size(0) == 3
    assert torch.equal(batch.seed_index, torch.tensor([1, 2]))
    assert torch.equal(
        batch.graph.edges[("author", "writes", "paper")].edge_index,
        torch.tensor([[0, 2], [1, 2]]),
    )
