from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vgl.dataloading import DataLoader, ListDataset, NodeNeighborSampler
from vgl.engine import Trainer
from vgl.graph import Graph
from vgl.tasks import NodeClassificationTask


class TinyHeteroModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, graph):
        if hasattr(graph, "graph"):
            graph = graph.graph
        return self.linear(graph.nodes["paper"].x)


def build_demo_graph():
    return Graph.hetero(
        nodes={
            "paper": {
                "x": torch.randn(3, 4),
                "y": torch.tensor([0, 1, 0]),
                "train_mask": torch.tensor([True, True, False]),
                "val_mask": torch.tensor([False, True, True]),
                "test_mask": torch.tensor([True, False, True]),
            },
            "author": {"x": torch.randn(2, 4)},
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1], [1, 2]])
            },
            ("paper", "written_by", "author"): {
                "edge_index": torch.tensor([[1, 2], [0, 1]])
            },
        },
    )


def _seed_dataset(graph, mask_key):
    seeds = graph.nodes["paper"].data[mask_key].nonzero(as_tuple=False).view(-1).tolist()
    return ListDataset(
        [
            (graph, {"seed": int(seed), "node_type": "paper", "sample_id": f"{mask_key}:{seed}"})
            for seed in seeds
        ]
    )


def build_demo_loaders():
    graph = build_demo_graph()
    train_loader = DataLoader(
        dataset=_seed_dataset(graph, "train_mask"),
        sampler=NodeNeighborSampler(num_neighbors=[-1]),
        batch_size=2,
    )
    val_loader = DataLoader(
        dataset=_seed_dataset(graph, "val_mask"),
        sampler=NodeNeighborSampler(num_neighbors=[-1]),
        batch_size=2,
    )
    test_loader = DataLoader(
        dataset=_seed_dataset(graph, "test_mask"),
        sampler=NodeNeighborSampler(num_neighbors=[-1]),
        batch_size=2,
    )
    return train_loader, val_loader, test_loader


def main():
    task = NodeClassificationTask(
        target="y",
        split=("train_mask", "val_mask", "test_mask"),
        node_type="paper",
    )
    trainer = Trainer(model=TinyHeteroModel(), task=task, optimizer=torch.optim.Adam, lr=1e-2, max_epochs=1)
    train_loader, val_loader, test_loader = build_demo_loaders()
    result = {
        "history": trainer.fit(train_loader, val_data=val_loader),
        "test": trainer.test(test_loader),
    }
    print(result)


if __name__ == "__main__":
    main()
