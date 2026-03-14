import pytest
import torch

from vgl import Graph
from vgl.data.dataset import ListDataset
from vgl.data.loader import Loader
from vgl.data.sample import LinkPredictionRecord
from vgl.data.sampler import FullGraphSampler


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


def test_loader_rejects_link_prediction_records_from_multiple_graphs():
    dataset = ListDataset(
        [
            LinkPredictionRecord(graph=_graph(), src_index=0, dst_index=1, label=1),
            LinkPredictionRecord(graph=_graph(), src_index=1, dst_index=2, label=0),
        ]
    )
    loader = Loader(dataset=dataset, sampler=FullGraphSampler(), batch_size=2)

    with pytest.raises(ValueError, match="single source graph"):
        next(iter(loader))
