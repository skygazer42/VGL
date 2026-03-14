import torch
from torch import nn

from vgl import Graph
from vgl.train.tasks import NodeClassificationTask
from vgl.train.trainer import Trainer


class TinyHeteroModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, graph):
        return self.linear(graph.nodes["paper"].x)


def test_end_to_end_hetero_training_runs():
    graph = Graph.hetero(
        nodes={
            "paper": {
                "x": torch.randn(2, 4),
                "y": torch.tensor([0, 1]),
                "train_mask": torch.tensor([True, True]),
                "val_mask": torch.tensor([True, True]),
                "test_mask": torch.tensor([True, True]),
            },
            "author": {
                "x": torch.randn(2, 4),
            },
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

    assert result["epochs"] == 1

