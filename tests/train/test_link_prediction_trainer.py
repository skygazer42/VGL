import pytest
import torch
from torch import nn

from vgl import Graph
from vgl.core.batch import LinkPredictionBatch
from vgl.data.dataset import ListDataset
from vgl.data.sample import LinkPredictionRecord
from vgl.data.sampler import CandidateLinkSampler
from vgl.data.sampler import FullGraphSampler
from vgl.data.sampler import LinkNeighborSampler
from vgl.data.sampler import HardNegativeLinkSampler
from vgl.data.sampler import UniformNegativeLinkSampler
from vgl.data.loader import Loader
from vgl.train.tasks import LinkPredictionTask
from vgl.train.trainer import Trainer


def _batch():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 4),
    )
    return LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1),
            LinkPredictionRecord(graph=graph, src_index=2, dst_index=0, label=0),
        ]
    )


class TinyLinkPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.scorer = nn.Linear(8, 1)

    def forward(self, batch):
        node_repr = self.encoder(batch.graph.x)
        src_x = node_repr[batch.src_index]
        dst_x = node_repr[batch.dst_index]
        return self.scorer(torch.cat([src_x, dst_x], dim=-1)).squeeze(-1)


class RecordingLinkPredictor(TinyLinkPredictor):
    def __init__(self):
        super().__init__()
        self.seen_edge_index = None

    def forward(self, batch):
        self.seen_edge_index = batch.graph.edge_index.detach().clone()
        return super().forward(batch)


def test_trainer_runs_link_prediction_epoch():
    trainer = Trainer(
        model=TinyLinkPredictor(),
        task=LinkPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit([_batch()])

    assert history["epochs"] == 1


def test_trainer_runs_link_prediction_epoch_with_uniform_negative_sampling():
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
    loader = Loader(
        dataset=dataset,
        sampler=UniformNegativeLinkSampler(num_negatives=2),
        batch_size=2,
    )
    trainer = Trainer(
        model=TinyLinkPredictor(),
        task=LinkPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["epochs"] == 1
    assert history["completed_epochs"] == 1


def test_trainer_receives_message_passing_graph_without_seed_edges():
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
    model = RecordingLinkPredictor()
    trainer = Trainer(
        model=model,
        task=LinkPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["completed_epochs"] == 1
    assert torch.equal(model.seen_edge_index, torch.tensor([[1, 1], [2, 0]]))


def test_trainer_runs_link_prediction_epoch_with_hard_negative_sampling():
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
    trainer = Trainer(
        model=TinyLinkPredictor(),
        task=LinkPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["epochs"] == 1
    assert history["completed_epochs"] == 1


def test_trainer_runs_link_prediction_epoch_with_link_neighbor_sampling():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        x=torch.randn(5, 4),
    )
    dataset = ListDataset(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1),
            LinkPredictionRecord(graph=graph, src_index=1, dst_index=2, label=1),
        ]
    )
    loader = Loader(
        dataset=dataset,
        sampler=LinkNeighborSampler(
            num_neighbors=[-1],
            base_sampler=UniformNegativeLinkSampler(num_negatives=1),
        ),
        batch_size=2,
    )
    trainer = Trainer(
        model=TinyLinkPredictor(),
        task=LinkPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["epochs"] == 1
    assert history["completed_epochs"] == 1


def test_trainer_runs_link_prediction_epoch_with_hetero_link_neighbor_sampling():
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
            LinkPredictionRecord(graph=graph, src_index=1, dst_index=2, label=1, edge_type=edge_type),
        ]
    )
    loader = Loader(
        dataset=dataset,
        sampler=LinkNeighborSampler(
            num_neighbors=[-1],
            base_sampler=UniformNegativeLinkSampler(num_negatives=1),
        ),
        batch_size=2,
    )

    class TinyHeteroLinkPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.author_encoder = nn.Linear(4, 4)
            self.paper_encoder = nn.Linear(4, 4)
            self.scorer = nn.Linear(8, 1)

        def forward(self, batch):
            author_x = self.author_encoder(batch.graph.nodes["author"].x)
            paper_x = self.paper_encoder(batch.graph.nodes["paper"].x)
            src_x = author_x[batch.src_index]
            dst_x = paper_x[batch.dst_index]
            return self.scorer(torch.cat([src_x, dst_x], dim=-1)).squeeze(-1)

    trainer = Trainer(
        model=TinyHeteroLinkPredictor(),
        task=LinkPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["epochs"] == 1
    assert history["completed_epochs"] == 1


def test_trainer_runs_link_prediction_epoch_with_mixed_hetero_edge_types():
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
    loader = Loader(
        dataset=ListDataset(
            [
                LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1, edge_type=writes),
                LinkPredictionRecord(graph=graph, src_index=2, dst_index=3, label=0, edge_type=cites),
            ]
        ),
        sampler=FullGraphSampler(),
        batch_size=2,
    )

    class TinyMixedHeteroLinkPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.author_encoder = nn.Linear(4, 4)
            self.paper_encoder = nn.Linear(4, 4)
            self.scorer = nn.Linear(8, 1)

        def forward(self, batch):
            author_x = self.author_encoder(batch.graph.nodes["author"].x)
            paper_x = self.paper_encoder(batch.graph.nodes["paper"].x)
            logits = torch.zeros(batch.labels.size(0), dtype=paper_x.dtype, device=paper_x.device)
            for index, relation_index in enumerate(batch.edge_type_index.tolist()):
                src_type, _, dst_type = batch.edge_types[relation_index]
                if src_type == "author":
                    src_repr = author_x[batch.src_index[index]]
                else:
                    src_repr = paper_x[batch.src_index[index]]
                if dst_type == "author":
                    dst_repr = author_x[batch.dst_index[index]]
                else:
                    dst_repr = paper_x[batch.dst_index[index]]
                logits[index] = self.scorer(torch.cat([src_repr, dst_repr], dim=-1)).squeeze(0)
            return logits

    trainer = Trainer(
        model=TinyMixedHeteroLinkPredictor(),
        task=LinkPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["epochs"] == 1
    assert history["completed_epochs"] == 1


def test_trainer_tracks_link_prediction_ranking_metrics():
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
        sampler=HardNegativeLinkSampler(num_negatives=2, num_hard_negatives=2),
        batch_size=1,
    )

    class RankingAwareLinkPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.tensor(0.0))

        def forward(self, batch):
            return -batch.dst_index.to(dtype=torch.float32) + self.bias

    trainer = Trainer(
        model=RankingAwareLinkPredictor(),
        task=LinkPredictionTask(target="label", metrics=["mrr", "hits@1", "hits@3"]),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["train"][0]["mrr"] == pytest.approx(1.0)
    assert history["train"][0]["hits@1"] == pytest.approx(1.0)
    assert history["train"][0]["hits@3"] == pytest.approx(1.0)


def test_trainer_tracks_filtered_link_prediction_ranking_metrics():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(4, 4),
    )
    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1, query_id="q0"),
            LinkPredictionRecord(
                graph=graph,
                src_index=0,
                dst_index=2,
                label=0,
                query_id="q0",
                filter_ranking=True,
            ),
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=3, label=0, query_id="q0"),
        ]
    )

    class FilterAwareLinkPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.tensor(0.0))

        def forward(self, batch):
            return torch.tensor([0.8, 0.9, 0.1], dtype=torch.float32) + self.bias

    trainer = Trainer(
        model=FilterAwareLinkPredictor(),
        task=LinkPredictionTask(target="label", metrics=["filtered_mrr", "filtered_hits@1"]),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit([batch])

    assert history["train"][0]["filtered_mrr"] == pytest.approx(1.0)
    assert history["train"][0]["filtered_hits@1"] == pytest.approx(1.0)


def test_trainer_tracks_raw_and_filtered_ranking_metrics_with_candidate_sampler():
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

    class CandidateAwareLinkPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.tensor(0.0))

        def forward(self, batch):
            scores = {
                0: 0.1,
                1: 0.8,
                2: 0.9,
                3: 0.0,
            }
            logits = [scores[int(dst)] for dst in batch.dst_index.tolist()]
            return torch.tensor(logits, dtype=torch.float32) + self.bias

    trainer = Trainer(
        model=CandidateAwareLinkPredictor(),
        task=LinkPredictionTask(target="label", metrics=["mrr", "filtered_mrr", "filtered_hits@1"]),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(loader)

    assert history["train"][0]["mrr"] == pytest.approx(0.5)
    assert history["train"][0]["filtered_mrr"] == pytest.approx(1.0)
    assert history["train"][0]["filtered_hits@1"] == pytest.approx(1.0)
