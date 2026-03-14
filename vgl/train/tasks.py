import torch
import torch.nn.functional as F

from vgl.train.task import Task


class NodeClassificationTask(Task):
    def __init__(self, target, split, loss="cross_entropy", metrics=None, node_type=None):
        self.target = target
        self.train_key, self.val_key, self.test_key = split
        self.loss_name = loss
        self.metrics = metrics or []
        self.node_type = node_type

    def _node_data(self, graph):
        if self.node_type is not None:
            return graph.nodes[self.node_type].data
        if "node" in graph.nodes:
            return graph.nodes["node"].data
        if len(graph.nodes) == 1:
            return next(iter(graph.nodes.values())).data
        raise ValueError("node_type is required for multi-type node classification")

    def loss(self, graph, logits, stage):
        return F.cross_entropy(
            self.predictions_for_metrics(graph, logits, stage),
            self.targets(graph, stage),
        )

    def targets(self, graph, stage):
        node_data = self._node_data(graph)
        mask = node_data[f"{stage}_mask"]
        target = node_data[self.target]
        return target[mask]

    def predictions_for_metrics(self, graph, predictions, stage):
        node_data = self._node_data(graph)
        mask = node_data[f"{stage}_mask"]
        return predictions[mask]


class GraphClassificationTask(Task):
    def __init__(self, target, label_source="graph", loss="cross_entropy", metrics=None):
        self.target = target
        self.label_source = label_source
        self.loss_name = loss
        self.metrics = metrics or []

    def _targets(self, batch):
        if self.label_source == "graph":
            return batch.labels
        if self.label_source == "metadata":
            return batch.labels if getattr(batch, "labels", None) is not None else torch.tensor(
                [item[self.target] for item in batch.metadata]
            )
        if self.label_source == "auto":
            if getattr(batch, "labels", None) is not None:
                return batch.labels
            return torch.tensor([item[self.target] for item in batch.metadata])
        raise ValueError(f"Unsupported label_source: {self.label_source}")

    def loss(self, batch, logits, stage):
        del stage
        target = self._targets(batch)
        return F.cross_entropy(logits, target)

    def targets(self, batch, stage):
        del stage
        return self._targets(batch)


class LinkPredictionTask(Task):
    def __init__(self, target="label", loss="binary_cross_entropy", metrics=None):
        if loss != "binary_cross_entropy":
            raise ValueError(f"Unsupported loss: {loss}")
        self.target = target
        self.loss_name = loss
        self.metrics = metrics or []

    def loss(self, batch, logits, stage):
        logits = self.predictions_for_metrics(batch, logits, stage=stage)
        targets = self.targets(batch, stage=stage).to(dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets)

    def predictions_for_metrics(self, batch, predictions, stage):
        del batch, stage
        if predictions.ndim == 2 and predictions.size(-1) == 1:
            predictions = predictions.squeeze(-1)
        if predictions.ndim != 1:
            raise ValueError("LinkPredictionTask expects one logit per candidate edge")
        return predictions

    def targets(self, batch, stage):
        del stage
        return batch.labels


class TemporalEventPredictionTask(Task):
    def __init__(self, target="label", loss="cross_entropy", metrics=None):
        if loss != "cross_entropy":
            raise ValueError(f"Unsupported loss: {loss}")
        self.target = target
        self.loss_name = loss
        self.metrics = metrics or []

    def loss(self, batch, logits, stage):
        del stage
        return F.cross_entropy(logits, batch.labels)

    def targets(self, batch, stage):
        del stage
        return batch.labels

