import torch.nn.functional as F

from gnn.train.task import Task


class NodeClassificationTask(Task):
    def __init__(self, target, split, loss="cross_entropy", metrics=None):
        self.target = target
        self.train_key, self.val_key, self.test_key = split
        self.loss_name = loss
        self.metrics = metrics or []

    def loss(self, graph, logits, stage):
        mask = getattr(graph, f"{stage}_mask")
        target = getattr(graph, self.target)
        return F.cross_entropy(logits[mask], target[mask])
