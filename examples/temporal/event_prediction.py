from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gnn import Graph
from gnn.train.tasks import NodeClassificationTask
from gnn.train.trainer import Trainer


class TinyTemporalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, graph):
        return self.linear(graph.x)


def main():
    graph = Graph.temporal(
        nodes={
            "node": {
                "x": torch.randn(2, 4),
                "y": torch.tensor([0, 1]),
                "train_mask": torch.tensor([True, True]),
                "val_mask": torch.tensor([True, True]),
                "test_mask": torch.tensor([True, True]),
            }
        },
        edges={
            ("node", "interacts", "node"): {
                "edge_index": torch.tensor([[0, 1], [1, 0]]),
                "timestamp": torch.tensor([1, 2]),
            }
        },
        time_attr="timestamp",
    )
    task = NodeClassificationTask(target="y", split=("train_mask", "val_mask", "test_mask"))
    trainer = Trainer(model=TinyTemporalModel(), task=task, optimizer=torch.optim.Adam, lr=1e-2, max_epochs=1)
    result = trainer.fit(graph)
    print(result)


if __name__ == "__main__":
    main()
