import torch
from torch import nn

from vgl import Graph
from vgl.core.batch import TemporalEventBatch
from vgl.data.sample import TemporalEventRecord
from vgl.data.dataset import ListDataset
from vgl.data.loader import Loader
from vgl.data.sampler import TemporalNeighborSampler
from vgl.train.tasks import TemporalEventPredictionTask
from vgl.train.trainer import Trainer


def _batch():
    graph = Graph.temporal(
        nodes={"node": {"x": torch.randn(3, 4)}},
        edges={
            ("node", "interacts", "node"): {
                "edge_index": torch.tensor([[0, 1], [1, 2]]),
                "timestamp": torch.tensor([1, 4]),
            }
        },
        time_attr="timestamp",
    )
    return TemporalEventBatch.from_records(
        [
            TemporalEventRecord(graph=graph, src_index=0, dst_index=1, timestamp=1, label=1),
            TemporalEventRecord(graph=graph, src_index=1, dst_index=2, timestamp=4, label=0),
        ]
    )


class TinyTemporalEventModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(9, 2)

    def forward(self, batch):
        src_x = batch.graph.x[batch.src_index]
        dst_x = batch.graph.x[batch.dst_index]
        history_counts = torch.tensor(
            [batch.history_graph(i).edge_index.size(1) for i in range(batch.labels.size(0))],
            dtype=src_x.dtype,
        ).unsqueeze(-1)
        return self.linear(torch.cat([src_x, dst_x, history_counts], dim=-1))


def test_trainer_runs_temporal_event_prediction_epoch():
    batch = _batch()
    trainer = Trainer(
        model=TinyTemporalEventModel(),
        task=TemporalEventPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit([batch])

    assert history["epochs"] == 1


def test_trainer_runs_temporal_event_prediction_epoch_with_temporal_neighbor_sampling():
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
    loader = Loader(
        dataset=ListDataset(
            [
                TemporalEventRecord(graph=graph, src_index=1, dst_index=2, timestamp=3, label=1),
                TemporalEventRecord(graph=graph, src_index=2, dst_index=0, timestamp=6, label=0),
            ]
        ),
        sampler=TemporalNeighborSampler(num_neighbors=[-1]),
        batch_size=2,
    )
    trainer = Trainer(
        model=TinyTemporalEventModel(),
        task=TemporalEventPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["epochs"] == 1

HETERO_EDGE_TYPE = ("author", "writes", "paper")


class TinyHeteroTemporalEventModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(9, 2)

    def forward(self, batch):
        src_x = batch.graph.nodes[batch.src_node_type].x[batch.src_index]
        dst_x = batch.graph.nodes[batch.dst_node_type].x[batch.dst_index]
        history_counts = torch.tensor(
            [batch.history_graph(i).edges[batch.edge_type].edge_index.size(1) for i in range(batch.labels.size(0))],
            dtype=src_x.dtype,
        ).unsqueeze(-1)
        return self.linear(torch.cat([src_x, dst_x, history_counts], dim=-1))


def test_trainer_runs_temporal_event_prediction_epoch_with_hetero_temporal_neighbor_sampling():
    graph = Graph.temporal(
        nodes={
            "author": {"x": torch.randn(2, 4)},
            "paper": {"x": torch.randn(3, 4)},
        },
        edges={
            HETERO_EDGE_TYPE: {
                "edge_index": torch.tensor([[0, 1, 1], [1, 0, 2]]),
                "timestamp": torch.tensor([1, 4, 6]),
            }
        },
        time_attr="timestamp",
    )
    loader = Loader(
        dataset=ListDataset(
            [
                TemporalEventRecord(graph=graph, src_index=1, dst_index=2, timestamp=5, label=1, edge_type=HETERO_EDGE_TYPE),
                TemporalEventRecord(graph=graph, src_index=0, dst_index=1, timestamp=2, label=0, edge_type=HETERO_EDGE_TYPE),
            ]
        ),
        sampler=TemporalNeighborSampler(num_neighbors=[-1]),
        batch_size=2,
    )
    trainer = Trainer(
        model=TinyHeteroTemporalEventModel(),
        task=TemporalEventPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["epochs"] == 1

