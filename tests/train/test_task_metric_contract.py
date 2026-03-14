import torch

from vgl import Graph
from vgl.core.batch import LinkPredictionBatch, TemporalEventBatch
from vgl.data.sample import LinkPredictionRecord, TemporalEventRecord
from vgl.train.tasks import (
    GraphClassificationTask,
    LinkPredictionTask,
    NodeClassificationTask,
    TemporalEventPredictionTask,
)


def test_node_classification_task_masks_metric_predictions_and_targets():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(3, 4),
        y=torch.tensor([0, 1, 1]),
        train_mask=torch.tensor([True, False, True]),
        val_mask=torch.tensor([True, True, True]),
        test_mask=torch.tensor([True, True, True]),
    )
    task = NodeClassificationTask(target="y", split=("train_mask", "val_mask", "test_mask"))
    logits = torch.randn(3, 2)

    preds = task.predictions_for_metrics(graph, logits, stage="train")
    targets = task.targets(graph, stage="train")

    assert preds.shape == (2, 2)
    assert torch.equal(targets, torch.tensor([0, 1]))


def test_graph_classification_task_exposes_graph_targets():
    class FakeBatch:
        labels = torch.tensor([1, 0])
        metadata = [{"label": 1}, {"label": 0}]

    task = GraphClassificationTask(target="label", label_source="metadata")

    assert torch.equal(task.targets(FakeBatch(), stage="train"), torch.tensor([1, 0]))


def test_link_prediction_task_exposes_metric_ready_predictions_and_targets():
    graph = Graph.homo(edge_index=torch.tensor([[0], [1]]), x=torch.randn(2, 4))
    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1),
            LinkPredictionRecord(graph=graph, src_index=1, dst_index=0, label=0),
        ]
    )
    task = LinkPredictionTask()
    logits = torch.randn(2, 1)

    preds = task.predictions_for_metrics(batch, logits, stage="train")

    assert preds.shape == (2,)
    assert torch.equal(task.targets(batch, stage="train"), torch.tensor([1.0, 0.0]))


def test_temporal_event_task_exposes_targets():
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
    batch = TemporalEventBatch.from_records(
        [
            TemporalEventRecord(graph=graph, src_index=0, dst_index=1, timestamp=1, label=1),
            TemporalEventRecord(graph=graph, src_index=1, dst_index=2, timestamp=4, label=0),
        ]
    )
    task = TemporalEventPredictionTask()

    assert torch.equal(task.targets(batch, stage="train"), torch.tensor([1, 0]))
