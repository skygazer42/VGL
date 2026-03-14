import torch

from vgl import Graph
from vgl.train.tasks import NodeClassificationTask


def test_node_classification_task_computes_loss():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
        train_mask=torch.tensor([True, True]),
    )
    task = NodeClassificationTask(
        target="y",
        split=("train_mask", "train_mask", "train_mask"),
    )
    logits = torch.randn(2, 2, requires_grad=True)

    loss = task.loss(graph, logits, stage="train")

    assert loss.ndim == 0

