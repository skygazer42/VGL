import pytest
import torch

from gnn import Graph
from gnn.data.dataset import ListDataset
from gnn.data.loader import Loader
from gnn.data.sample import TemporalEventRecord
from gnn.data.sampler import FullGraphSampler


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
            TemporalEventRecord(graph=graph, src_index=0, dst_index=1, timestamp=1, label=1),
            TemporalEventRecord(graph=graph, src_index=1, dst_index=2, timestamp=4, label=0),
        ]
    )
    loader = Loader(dataset=dataset, sampler=FullGraphSampler(), batch_size=2)

    batch = next(iter(loader))

    assert torch.equal(batch.labels, torch.tensor([1, 0]))
    assert torch.equal(batch.timestamp, torch.tensor([1, 4]))


def test_loader_rejects_temporal_records_from_multiple_graphs():
    dataset = ListDataset(
        [
            TemporalEventRecord(graph=_temporal_graph(), src_index=0, dst_index=1, timestamp=1, label=1),
            TemporalEventRecord(graph=_temporal_graph(), src_index=1, dst_index=2, timestamp=4, label=0),
        ]
    )
    loader = Loader(dataset=dataset, sampler=FullGraphSampler(), batch_size=2)

    with pytest.raises(ValueError, match="single source graph"):
        next(iter(loader))
