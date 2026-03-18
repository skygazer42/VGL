import pytest
import torch

from vgl import Graph
from vgl.data.dataset import ListDataset
from vgl.data.loader import Loader
from vgl.data.sample import LinkPredictionRecord
from vgl.data.sampler import CandidateLinkSampler
from vgl.data.sampler import HardNegativeLinkSampler
from vgl.data.sampler import FullGraphSampler
from vgl.data.sampler import UniformNegativeLinkSampler


def _graph():
    return Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 4),
    )


def test_loader_collates_link_prediction_records():
    graph = _graph()
    dataset = ListDataset(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1),
            LinkPredictionRecord(graph=graph, src_index=2, dst_index=0, label=0),
        ]
    )
    loader = Loader(dataset=dataset, sampler=FullGraphSampler(), batch_size=2)

    batch = next(iter(loader))

    assert batch.graph is graph
    assert torch.equal(batch.labels, torch.tensor([1.0, 0.0]))


def test_loader_batches_link_prediction_records_from_multiple_graphs():
    dataset = ListDataset(
        [
            LinkPredictionRecord(graph=_graph(), src_index=0, dst_index=1, label=1),
            LinkPredictionRecord(graph=_graph(), src_index=1, dst_index=2, label=0),
        ]
    )
    loader = Loader(dataset=dataset, sampler=FullGraphSampler(), batch_size=2)

    batch = next(iter(loader))

    assert torch.equal(batch.src_index, torch.tensor([0, 4]))
    assert torch.equal(batch.dst_index, torch.tensor([1, 5]))
    assert batch.graph.x.size(0) == 6


def test_loader_can_expand_positive_link_records_with_uniform_negative_sampler():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(4, 4),
    )
    dataset = ListDataset(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1),
            LinkPredictionRecord(graph=graph, src_index=1, dst_index=2, label=1),
        ]
    )
    loader = Loader(dataset=dataset, sampler=UniformNegativeLinkSampler(num_negatives=2), batch_size=2)

    batch = next(iter(loader))
    positives = {
        (int(src), int(dst))
        for src, dst, label in zip(batch.src_index.tolist(), batch.dst_index.tolist(), batch.labels.tolist())
        if label == 1.0
    }
    negatives = [
        (int(src), int(dst))
        for src, dst, label in zip(batch.src_index.tolist(), batch.dst_index.tolist(), batch.labels.tolist())
        if label == 0.0
    ]

    assert batch.graph is graph
    assert torch.equal(batch.labels, torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
    assert torch.equal(batch.query_index, torch.tensor([0, 0, 0, 1, 1, 1]))
    assert positives == {(0, 1), (1, 2)}
    assert len(negatives) == 4
    assert all(edge not in {(0, 1), (1, 2)} for edge in negatives)


def test_loader_can_expand_positive_hetero_link_records_with_uniform_negative_sampler():
    edge_type = ("author", "writes", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(2, 4)},
            "paper": {"x": torch.randn(4, 4)},
        },
        edges={
            edge_type: {"edge_index": torch.tensor([[0, 1], [1, 2]])},
            ("paper", "written_by", "author"): {"edge_index": torch.tensor([[1, 2], [0, 1]])},
        },
    )
    dataset = ListDataset(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1, edge_type=edge_type),
        ]
    )
    loader = Loader(
        dataset=dataset,
        sampler=UniformNegativeLinkSampler(num_negatives=2),
        batch_size=1,
    )

    batch = next(iter(loader))

    assert batch.edge_type == edge_type
    assert batch.src_node_type == "author"
    assert batch.dst_node_type == "paper"
    assert torch.equal(batch.labels, torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(batch.query_index, torch.tensor([0, 0, 0]))
    assert torch.equal(batch.src_index, torch.tensor([0, 0, 0]))
    assert all(0 <= int(dst) < graph.nodes["paper"].x.size(0) for dst in batch.dst_index.tolist())
    assert all(int(dst) != 1 for dst in batch.dst_index[1:].tolist())


def test_loader_batches_mixed_hetero_edge_types_in_one_link_batch():
    writes = ("author", "writes", "paper")
    cites = ("paper", "cites", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(2, 4)},
            "paper": {"x": torch.randn(4, 4)},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1], [1, 2]])},
            ("paper", "written_by", "author"): {"edge_index": torch.tensor([[1, 2], [0, 1]])},
            cites: {"edge_index": torch.tensor([[0, 2], [2, 3]])},
        },
    )
    dataset = ListDataset(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1, edge_type=writes),
            LinkPredictionRecord(graph=graph, src_index=2, dst_index=3, label=0, edge_type=cites),
        ]
    )
    loader = Loader(dataset=dataset, sampler=FullGraphSampler(), batch_size=2)

    batch = next(iter(loader))

    assert batch.edge_type is None
    assert batch.edge_types == (writes, cites)
    assert torch.equal(batch.edge_type_index, torch.tensor([0, 1]))
    assert torch.equal(batch.src_index, torch.tensor([0, 2]))
    assert torch.equal(batch.dst_index, torch.tensor([1, 3]))


def test_loader_can_uniform_negative_sample_mixed_hetero_edge_types():
    writes = ("author", "writes", "paper")
    cites = ("paper", "cites", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(2, 4)},
            "paper": {"x": torch.randn(4, 4)},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1], [1, 2]])},
            ("paper", "written_by", "author"): {"edge_index": torch.tensor([[1, 2], [0, 1]])},
            cites: {"edge_index": torch.tensor([[0, 2], [2, 3]])},
        },
    )
    dataset = ListDataset(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1, edge_type=writes),
            LinkPredictionRecord(graph=graph, src_index=2, dst_index=3, label=1, edge_type=cites),
        ]
    )
    loader = Loader(
        dataset=dataset,
        sampler=UniformNegativeLinkSampler(num_negatives=1),
        batch_size=2,
    )

    batch = next(iter(loader))

    assert batch.edge_type is None
    assert batch.edge_types == (writes, cites)
    assert torch.equal(batch.edge_type_index, torch.tensor([0, 0, 1, 1]))
    assert torch.equal(batch.labels, torch.tensor([1.0, 0.0, 1.0, 0.0]))


def test_uniform_negative_link_sampler_requires_positive_seed_records():
    graph = _graph()
    dataset = ListDataset(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=0),
        ]
    )
    loader = Loader(dataset=dataset, sampler=UniformNegativeLinkSampler(num_negatives=1), batch_size=1)

    with pytest.raises(ValueError, match="positive"):
        next(iter(loader))


def test_loader_can_expand_positive_link_records_with_hard_negative_sampler():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1], [1, 2, 2]]),
        x=torch.randn(5, 4),
    )
    dataset = ListDataset(
        [
            LinkPredictionRecord(
                graph=graph,
                src_index=0,
                dst_index=1,
                label=1,
                hard_negative_dst=[3, 4],
            ),
        ]
    )
    loader = Loader(
        dataset=dataset,
        sampler=HardNegativeLinkSampler(num_negatives=3, num_hard_negatives=2),
        batch_size=1,
    )

    batch = next(iter(loader))
    negatives = [
        (int(src), int(dst))
        for src, dst, label in zip(batch.src_index.tolist(), batch.dst_index.tolist(), batch.labels.tolist())
        if label == 0.0
    ]

    assert torch.equal(batch.labels, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert {(0, 3), (0, 4)}.issubset(set(negatives))
    assert len(negatives) == 3
    assert all(edge not in {(0, 1), (0, 2)} for edge in negatives)


def test_loader_can_expand_positive_link_records_with_candidate_sampler_using_all_destinations():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1], [1, 2, 2]]),
        x=torch.randn(4, 4),
    )
    dataset = ListDataset(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1),
        ]
    )
    loader = Loader(
        dataset=dataset,
        sampler=CandidateLinkSampler(),
        batch_size=1,
    )

    batch = next(iter(loader))

    assert torch.equal(batch.src_index, torch.tensor([0, 0, 0, 0]))
    assert torch.equal(batch.dst_index, torch.tensor([1, 0, 2, 3]))
    assert torch.equal(batch.labels, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.equal(batch.query_index, torch.tensor([0, 0, 0, 0]))
    assert torch.equal(batch.filter_mask, torch.tensor([False, False, True, False]))


def test_candidate_link_sampler_supports_explicit_candidate_sets():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0], [1, 2]]),
        x=torch.randn(5, 4),
    )
    dataset = ListDataset(
        [
            LinkPredictionRecord(
                graph=graph,
                src_index=0,
                dst_index=1,
                label=1,
                candidate_dst=[3, 2, 3],
            ),
        ]
    )
    loader = Loader(
        dataset=dataset,
        sampler=CandidateLinkSampler(),
        batch_size=1,
    )

    batch = next(iter(loader))

    assert torch.equal(batch.dst_index, torch.tensor([1, 3, 2]))
    assert torch.equal(batch.labels, torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(batch.filter_mask, torch.tensor([False, False, True]))


def test_candidate_link_sampler_rejects_out_of_range_candidates():
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.randn(3, 4),
    )
    dataset = ListDataset(
        [
            LinkPredictionRecord(
                graph=graph,
                src_index=0,
                dst_index=1,
                label=1,
                candidate_dst=[2, 4],
            ),
        ]
    )
    loader = Loader(
        dataset=dataset,
        sampler=CandidateLinkSampler(),
        batch_size=1,
    )

    with pytest.raises(ValueError, match="candidate_dst"):
        next(iter(loader))


def test_hard_negative_link_sampler_can_exclude_seed_edges_from_message_passing_graph():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1], [1, 2, 0]]),
        x=torch.randn(4, 4),
    )
    dataset = ListDataset(
        [
            LinkPredictionRecord(
                graph=graph,
                src_index=0,
                dst_index=1,
                label=1,
                hard_negative_dst=[3],
            ),
        ]
    )
    loader = Loader(
        dataset=dataset,
        sampler=HardNegativeLinkSampler(
            num_negatives=1,
            num_hard_negatives=1,
            exclude_seed_edges=True,
        ),
        batch_size=1,
    )

    batch = next(iter(loader))

    assert batch.graph is not graph
    assert torch.equal(batch.graph.edge_index, torch.tensor([[1, 1], [2, 0]]))
    assert torch.equal(batch.labels, torch.tensor([1.0, 0.0]))


def test_hard_negative_link_sampler_rejects_out_of_range_candidates():
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.randn(3, 4),
    )
    dataset = ListDataset(
        [
            LinkPredictionRecord(
                graph=graph,
                src_index=0,
                dst_index=1,
                label=1,
                hard_negative_dst=[4],
            ),
        ]
    )
    loader = Loader(
        dataset=dataset,
        sampler=HardNegativeLinkSampler(num_negatives=1, num_hard_negatives=1),
        batch_size=1,
    )

    with pytest.raises(ValueError, match="hard_negative_dst"):
        next(iter(loader))


def test_loader_can_exclude_seed_edges_from_message_passing_graph():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1], [1, 2, 0]]),
        x=torch.randn(3, 4),
    )
    dataset = ListDataset(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1),
        ]
    )
    loader = Loader(
        dataset=dataset,
        sampler=UniformNegativeLinkSampler(num_negatives=1, exclude_seed_edges=True),
        batch_size=1,
    )

    batch = next(iter(loader))

    assert batch.graph is not graph
    assert torch.equal(batch.graph.edge_index, torch.tensor([[1, 1], [2, 0]]))
    assert torch.equal(batch.labels, torch.tensor([1.0, 0.0]))
