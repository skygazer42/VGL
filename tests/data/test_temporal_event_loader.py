import torch

from vgl import Graph
from vgl.data.dataset import ListDataset
from vgl.data.loader import Loader
from vgl.data.sample import TemporalEventRecord
from vgl.data.sampler import FullGraphSampler
from vgl.data.sampler import TemporalNeighborSampler


def _temporal_graph():
    return Graph.temporal(
        nodes={"node": {"x": torch.randn(3, 4)}},
        edges={
            ("node", "interacts", "node"): {
                "edge_index": torch.tensor([[0, 1], [1, 2]]),
                "timestamp": torch.tensor([1, 4]),
            }
        },
        time_attr="timestamp",
    )


def test_loader_collates_temporal_event_records():
    graph = _temporal_graph()
    dataset = ListDataset(
        [
            TemporalEventRecord(
                graph=graph,
                src_index=0,
                dst_index=1,
                timestamp=1,
                label=1,
                event_features=torch.tensor([1.0, 0.0]),
            ),
            TemporalEventRecord(
                graph=graph,
                src_index=1,
                dst_index=2,
                timestamp=4,
                label=0,
                event_features=torch.tensor([0.0, 1.0]),
            ),
        ]
    )
    loader = Loader(dataset=dataset, sampler=FullGraphSampler(), batch_size=2)

    batch = next(iter(loader))

    assert torch.equal(batch.labels, torch.tensor([1, 0]))
    assert torch.equal(batch.timestamp, torch.tensor([1, 4]))
    assert torch.equal(batch.event_features, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))


def test_loader_batches_temporal_records_from_multiple_graphs():
    dataset = ListDataset(
        [
            TemporalEventRecord(graph=_temporal_graph(), src_index=0, dst_index=1, timestamp=1, label=1),
            TemporalEventRecord(graph=_temporal_graph(), src_index=1, dst_index=2, timestamp=4, label=0),
        ]
    )
    loader = Loader(dataset=dataset, sampler=FullGraphSampler(), batch_size=2)

    batch = next(iter(loader))

    assert torch.equal(batch.src_index, torch.tensor([0, 4]))
    assert torch.equal(batch.dst_index, torch.tensor([1, 5]))
    assert batch.graph.x.size(0) == 6


def test_loader_can_build_temporal_event_batches_from_temporal_neighbor_sampler():
    graph = Graph.temporal(
        nodes={"node": {"x": torch.randn(4, 4)}},
        edges={
            ("node", "interacts", "node"): {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
                "timestamp": torch.tensor([1, 3, 5]),
            }
        },
        time_attr="timestamp",
    )
    dataset = ListDataset(
        [
            TemporalEventRecord(graph=graph, src_index=1, dst_index=2, timestamp=3, label=1),
            TemporalEventRecord(graph=graph, src_index=2, dst_index=0, timestamp=6, label=0),
        ]
    )
    loader = Loader(dataset=dataset, sampler=TemporalNeighborSampler(num_neighbors=[-1]), batch_size=2)

    batch = next(iter(loader))

    assert torch.equal(batch.timestamp, torch.tensor([3, 6]))
    assert batch.graph.x.size(0) == 6
