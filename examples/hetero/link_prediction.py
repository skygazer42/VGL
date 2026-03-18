from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vgl.dataloading import (
    CandidateLinkSampler,
    DataLoader,
    LinkNeighborSampler,
    LinkPredictionRecord,
    ListDataset,
    UniformNegativeLinkSampler,
)
from vgl.engine import Trainer
from vgl.graph import Graph
from vgl.tasks import LinkPredictionTask


EDGE_TYPE = ("author", "writes", "paper")
REVERSE_EDGE_TYPE = ("paper", "written_by", "author")
CITES_EDGE_TYPE = ("paper", "cites", "paper")


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


class TinyMixedHeteroLinkPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.author_encoder = nn.Linear(4, 4)
        self.paper_encoder = nn.Linear(4, 4)
        self.scorer = nn.Linear(8, 1)

    def _node_repr(self, node_type, author_x, paper_x):
        if node_type == "author":
            return author_x
        if node_type == "paper":
            return paper_x
        raise ValueError(f"Unsupported node_type: {node_type}")

    def forward(self, batch):
        author_x = self.author_encoder(batch.graph.nodes["author"].x)
        paper_x = self.paper_encoder(batch.graph.nodes["paper"].x)
        if batch.edge_types is None or batch.edge_type_index is None:
            src_x = author_x[batch.src_index]
            dst_x = paper_x[batch.dst_index]
            return self.scorer(torch.cat([src_x, dst_x], dim=-1)).squeeze(-1)

        logits = torch.zeros(batch.labels.size(0), dtype=paper_x.dtype, device=paper_x.device)
        for index, relation_index in enumerate(batch.edge_type_index.tolist()):
            src_type, _, dst_type = batch.edge_types[relation_index]
            src_repr = self._node_repr(src_type, author_x, paper_x)[batch.src_index[index]]
            dst_repr = self._node_repr(dst_type, author_x, paper_x)[batch.dst_index[index]]
            logits[index] = self.scorer(torch.cat([src_repr, dst_repr], dim=-1)).squeeze(0)
        return logits


def build_demo_graph():
    return Graph.hetero(
        nodes={
            "author": {"x": torch.randn(3, 4)},
            "paper": {"x": torch.randn(4, 4)},
        },
        edges={
            EDGE_TYPE: {"edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]])},
            REVERSE_EDGE_TYPE: {"edge_index": torch.tensor([[1, 2, 3], [0, 1, 2]])},
            CITES_EDGE_TYPE: {"edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]])},
        },
    )


def _record(src_index, dst_index, *, label=1, candidate_dst=None, sample_id=None):
    return LinkPredictionRecord(
        graph=build_demo_graph(),
        src_index=src_index,
        dst_index=dst_index,
        label=label,
        edge_type=EDGE_TYPE,
        candidate_dst=candidate_dst,
        sample_id=sample_id,
    )


def build_demo_loaders():
    train_loader = DataLoader(
        dataset=ListDataset(
            [
                _record(0, 1, sample_id="train:0"),
                _record(1, 2, sample_id="train:1"),
            ]
        ),
        sampler=LinkNeighborSampler(
            num_neighbors=[-1],
            base_sampler=UniformNegativeLinkSampler(num_negatives=1),
        ),
        batch_size=2,
    )
    val_loader = DataLoader(
        dataset=ListDataset(
            [
                _record(2, 3, candidate_dst=[0, 2, 3], sample_id="val:0"),
            ]
        ),
        sampler=LinkNeighborSampler(
            num_neighbors=[-1],
            base_sampler=CandidateLinkSampler(),
        ),
        batch_size=1,
    )
    test_loader = DataLoader(
        dataset=ListDataset(
            [
                _record(0, 1, candidate_dst=[0, 1, 3], sample_id="test:0"),
            ]
        ),
        sampler=LinkNeighborSampler(
            num_neighbors=[-1],
            base_sampler=CandidateLinkSampler(),
        ),
        batch_size=1,
    )
    return train_loader, val_loader, test_loader


def build_mixed_demo_loaders():
    graph = build_demo_graph()
    train_loader = DataLoader(
        dataset=ListDataset(
            [
                LinkPredictionRecord(
                    graph=graph,
                    src_index=0,
                    dst_index=1,
                    label=1,
                    edge_type=EDGE_TYPE,
                    sample_id="train:writes:0",
                ),
                LinkPredictionRecord(
                    graph=graph,
                    src_index=1,
                    dst_index=2,
                    label=1,
                    edge_type=CITES_EDGE_TYPE,
                    sample_id="train:cites:0",
                ),
            ]
        ),
        sampler=LinkNeighborSampler(
            num_neighbors=[-1],
            base_sampler=UniformNegativeLinkSampler(num_negatives=1),
        ),
        batch_size=2,
    )
    val_loader = DataLoader(
        dataset=ListDataset(
            [
                LinkPredictionRecord(
                    graph=graph,
                    src_index=2,
                    dst_index=3,
                    label=1,
                    edge_type=EDGE_TYPE,
                    candidate_dst=[0, 2, 3],
                    sample_id="val:writes:0",
                ),
                LinkPredictionRecord(
                    graph=graph,
                    src_index=0,
                    dst_index=1,
                    label=1,
                    edge_type=CITES_EDGE_TYPE,
                    candidate_dst=[1, 2, 3],
                    sample_id="val:cites:0",
                ),
            ]
        ),
        sampler=LinkNeighborSampler(
            num_neighbors=[-1],
            base_sampler=CandidateLinkSampler(),
        ),
        batch_size=2,
    )
    test_loader = DataLoader(
        dataset=ListDataset(
            [
                LinkPredictionRecord(
                    graph=graph,
                    src_index=0,
                    dst_index=1,
                    label=1,
                    edge_type=EDGE_TYPE,
                    candidate_dst=[1, 2, 3],
                    sample_id="test:writes:0",
                ),
                LinkPredictionRecord(
                    graph=graph,
                    src_index=1,
                    dst_index=2,
                    label=1,
                    edge_type=CITES_EDGE_TYPE,
                    candidate_dst=[0, 2, 3],
                    sample_id="test:cites:0",
                ),
            ]
        ),
        sampler=LinkNeighborSampler(
            num_neighbors=[-1],
            base_sampler=CandidateLinkSampler(),
        ),
        batch_size=2,
    )
    return train_loader, val_loader, test_loader


def main():
    train_loader, val_loader, test_loader = build_demo_loaders()
    trainer = Trainer(
        model=TinyHeteroLinkPredictor(),
        task=LinkPredictionTask(target="label", metrics=["mrr", "filtered_mrr", "filtered_hits@1"]),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )
    result = {
        "history": trainer.fit(train_loader, val_data=val_loader),
        "test": trainer.test(test_loader),
    }
    print(result)
    return result


if __name__ == "__main__":
    main()
