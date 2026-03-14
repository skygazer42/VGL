from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vgl import Graph
from vgl.train.tasks import NodeClassificationTask
from vgl.train.trainer import Trainer


class TinyHeteroModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, graph):
        return self.linear(graph.nodes["paper"].x)


def main():
    graph = Graph.hetero(
        nodes={
            "paper": {
                "x": torch.randn(2, 4),
                "y": torch.tensor([0, 1]),
                "train_mask": torch.tensor([True, True]),
                "val_mask": torch.tensor([True, True]),
                "test_mask": torch.tensor([True, True]),
            },
            "author": {"x": torch.randn(2, 4)},
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1], [1, 0]])
            }
        },
    )
    task = NodeClassificationTask(
        target="y",
        split=("train_mask", "val_mask", "test_mask"),
        node_type="paper",
    )
    trainer = Trainer(model=TinyHeteroModel(), task=task, optimizer=torch.optim.Adam, lr=1e-2, max_epochs=1)
    result = trainer.fit(graph)
    print(result)


if __name__ == "__main__":
    main()

