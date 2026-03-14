# Training Loop Phase 5 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a stable training-loop surface with epoch-level metrics, validation and test evaluation, best-model restoration, and optional best-checkpoint saving.

**Architecture:** Extend the existing generic `Trainer` instead of introducing a second lifecycle object. Add a real streaming `Metric` contract, teach `Task` how to expose metric-aligned predictions and targets, and let `Trainer` aggregate epoch summaries and monitor a single best value. Keep checkpoint scope limited to model weights only.

**Tech Stack:** Python 3.11+, PyTorch, pytest, ruff, mypy

**Execution Rules:** Use `@test-driven-development` for every code task, keep commits small, and use `@verification-before-completion` before claiming the phase is done.

---

### Task 1: Add Metric Primitives and `Accuracy`

**Files:**
- Modify: `vgl/train/metrics.py`
- Modify: `vgl/train/__init__.py`
- Modify: `vgl/__init__.py`
- Modify: `tests/test_package_exports.py`
- Test: `tests/train/test_metrics.py`

**Step 1: Write the failing metric tests**

```python
import pytest
import torch

from vgl.train.metrics import Accuracy, build_metric


def test_accuracy_handles_multiclass_logits():
    metric = Accuracy()

    metric.update(
        torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
        torch.tensor([1, 0]),
    )

    assert metric.compute() == 1.0


def test_accuracy_handles_binary_logits():
    metric = Accuracy()

    metric.update(
        torch.tensor([1.5, -0.5, 0.1]),
        torch.tensor([1, 0, 1]),
    )

    assert metric.compute() == 1.0


def test_accuracy_rejects_shape_mismatch():
    metric = Accuracy()

    with pytest.raises(ValueError, match="shape"):
        metric.update(torch.randn(2, 3), torch.tensor([1]))


def test_build_metric_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unsupported metric"):
        build_metric("f1")
```

Update `tests/test_package_exports.py` so the root package also exposes:

```python
from vgl import Accuracy

assert Accuracy.__name__ == "Accuracy"
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/train/test_metrics.py tests/test_package_exports.py -v`
Expected: `FAIL` because `Accuracy` and metric builders do not exist yet.

**Step 3: Write minimal implementation**

Update `vgl/train/metrics.py`:

```python
from copy import deepcopy

import torch


class Metric:
    name = "metric"

    def reset(self):
        raise NotImplementedError

    def update(self, predictions, targets):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError


class Accuracy(Metric):
    name = "accuracy"

    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, predictions, targets):
        if predictions.ndim == targets.ndim + 1 and predictions.size(-1) == 1:
            predictions = predictions.squeeze(-1)
        if predictions.ndim == targets.ndim + 1:
            predicted = predictions.argmax(dim=-1)
        elif predictions.ndim == targets.ndim:
            predicted = (predictions >= 0).to(dtype=targets.dtype)
        else:
            raise ValueError("Accuracy received incompatible prediction/target shape")
        if predicted.shape != targets.shape:
            raise ValueError("Accuracy received incompatible prediction/target shape")
        self.correct += int((predicted == targets).sum().item())
        self.total += int(targets.numel())

    def compute(self):
        if self.total == 0:
            raise ValueError("Accuracy requires at least one example before compute()")
        return self.correct / self.total


def build_metric(metric):
    if isinstance(metric, Metric):
        return deepcopy(metric)
    if metric == "accuracy":
        return Accuracy()
    raise ValueError(f"Unsupported metric: {metric}")
```

Export `Accuracy` from `vgl/train/__init__.py` and `vgl/__init__.py`.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/train/test_metrics.py tests/test_package_exports.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add vgl/train/metrics.py vgl/train/__init__.py vgl/__init__.py tests/train/test_metrics.py tests/test_package_exports.py
git commit -m "feat: add accuracy metric"
```

### Task 2: Add Task Metric Hooks Across the Current Task Family

**Files:**
- Modify: `vgl/train/task.py`
- Modify: `vgl/train/tasks.py`
- Test: `tests/train/test_task_metric_contract.py`

**Step 1: Write the failing task-contract tests**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_task_metric_contract.py -v`
Expected: `FAIL` because `Task` and current task implementations do not expose metric hooks yet.

**Step 3: Write minimal implementation**

Update `vgl/train/task.py`:

```python
class Task:
    def loss(self, graph, predictions, stage):
        raise NotImplementedError

    def targets(self, batch, stage):
        raise NotImplementedError

    def predictions_for_metrics(self, batch, predictions, stage):
        del batch, stage
        return predictions
```

Update `vgl/train/tasks.py`:

```python
class NodeClassificationTask(Task):
    ...

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
    ...

    def targets(self, batch, stage):
        del stage
        return self._targets(batch)


class LinkPredictionTask(Task):
    ...

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
    ...

    def targets(self, batch, stage):
        del stage
        return batch.labels
```

Reuse `LinkPredictionTask.predictions_for_metrics(...)` inside `loss(...)` so shape validation lives in one place.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/train/test_task_metric_contract.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add vgl/train/task.py vgl/train/tasks.py tests/train/test_task_metric_contract.py
git commit -m "feat: add task metric hooks"
```

### Task 3: Extend `Trainer` with Evaluation, History, Monitoring, and Best-State Handling

**Files:**
- Modify: `vgl/train/trainer.py`
- Test: `tests/train/test_trainer_evaluation.py`
- Modify: `tests/train/test_trainer.py`

**Step 1: Write the failing trainer tests**

Create `tests/train/test_trainer_evaluation.py`:

```python
from pathlib import Path

import pytest
import torch
from torch import nn

from vgl import Graph
from vgl.train.task import Task
from vgl.train.tasks import NodeClassificationTask
from vgl.train.trainer import Trainer


class TinyNodeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, graph):
        return self.linear(graph.x)


def _graph():
    return Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(3, 4),
        y=torch.tensor([0, 1, 1]),
        train_mask=torch.tensor([True, True, True]),
        val_mask=torch.tensor([True, True, True]),
        test_mask=torch.tensor([True, True, True]),
    )


def test_trainer_fit_returns_structured_history_and_metrics():
    task = NodeClassificationTask(
        target="y",
        split=("train_mask", "val_mask", "test_mask"),
        metrics=["accuracy"],
    )
    trainer = Trainer(
        model=TinyNodeModel(),
        task=task,
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=2,
    )

    history = trainer.fit(_graph(), val_data=_graph())

    assert history["epochs"] == 2
    assert len(history["train"]) == 2
    assert len(history["val"]) == 2
    assert "loss" in history["train"][0]
    assert "accuracy" in history["val"][0]
    assert history["monitor"] == "val_loss"


class ToyBatch:
    def __init__(self, target):
        self.target = torch.tensor([target], dtype=torch.float32)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.0]))

    def forward(self, batch):
        return self.weight.repeat(batch.target.size(0))


class ToyTask(Task):
    def loss(self, batch, predictions, stage):
        del stage
        return ((predictions - batch.target) ** 2).mean()

    def targets(self, batch, stage):
        del stage
        return batch.target


def test_evaluate_and_test_do_not_step_optimizer():
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
    )
    before = trainer.model.weight.detach().clone()

    trainer.evaluate([ToyBatch(1.0)], stage="val")
    trainer.test([ToyBatch(1.0)])

    after = trainer.model.weight.detach().clone()

    assert torch.equal(before, after)


def test_trainer_restores_best_state_and_saves_checkpoint(tmp_path):
    checkpoint = tmp_path / "best.pt"
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=3,
        monitor="val_loss",
        save_best_path=checkpoint,
    )

    history = trainer.fit([ToyBatch(2.0)], val_data=[ToyBatch(1.0)])
    saved_state = torch.load(checkpoint)

    assert history["best_epoch"] == 2
    assert torch.equal(trainer.model.weight.detach(), torch.tensor([0.0]))
    assert torch.equal(saved_state["weight"], torch.tensor([0.0]))


def test_trainer_rejects_val_monitor_without_val_data():
    trainer = Trainer(
        model=ToyModel(),
        task=ToyTask(),
        optimizer=torch.optim.SGD,
        lr=1.0,
        max_epochs=1,
        monitor="val_loss",
    )

    with pytest.raises(ValueError, match="val_data"):
        trainer.fit([ToyBatch(2.0)])
```

Update `tests/train/test_trainer.py` so it verifies the returned history keeps `epochs` but now also includes the `train` key:

```python
assert history["epochs"] == 1
assert len(history["train"]) == 1
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/train/test_trainer.py tests/train/test_trainer_evaluation.py -v`
Expected: `FAIL` because `Trainer` does not yet support evaluation, structured history, or monitoring.

**Step 3: Write minimal implementation**

Update `vgl/train/trainer.py`:

```python
from copy import deepcopy
from pathlib import Path
from collections.abc import Iterable

import torch

from vgl.train.metrics import build_metric


class Trainer:
    def __init__(
        self,
        model,
        task,
        optimizer,
        lr,
        max_epochs,
        metrics=None,
        monitor=None,
        monitor_mode=None,
        save_best_path=None,
    ):
        self.model = model
        self.task = task
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.max_epochs = max_epochs
        metric_specs = task.metrics if metrics is None else metrics
        self.metric_specs = list(metric_specs or [])
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.save_best_path = Path(save_best_path) if save_best_path is not None else None
        self.best_state_dict = None
        self.best_epoch = None
        self.best_metric = None

    def _metrics(self):
        return [build_metric(metric) for metric in self.metric_specs]

    def _run_epoch(self, data, stage, training):
        batches = list(self._batches(data))
        if not batches:
            raise ValueError(f"Trainer.{stage} requires at least one batch")
        metrics = self._metrics()
        for metric in metrics:
            metric.reset()

        total_loss = 0.0
        total_items = 0
        self.model.train(training)
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in batches:
                if training:
                    self.optimizer.zero_grad()
                predictions = self.model(batch)
                loss = self.task.loss(batch, predictions, stage=stage)
                metric_predictions = self.task.predictions_for_metrics(batch, predictions, stage=stage)
                targets = self.task.targets(batch, stage=stage)
                if metric_predictions.size(0) != targets.size(0):
                    raise ValueError("Task metric predictions and targets must align in batch size")
                if training:
                    loss.backward()
                    self.optimizer.step()
                count = int(targets.size(0))
                total_loss += float(loss.item()) * count
                total_items += count
                for metric in metrics:
                    metric.update(metric_predictions.detach(), targets.detach())

        summary = {"loss": total_loss / total_items}
        for metric in metrics:
            value = metric.compute()
            if not isinstance(value, (int, float)):
                raise ValueError(f"Metric {metric.name} must return a scalar value")
            summary[metric.name] = float(value)
        return summary

    def _resolve_monitor(self, val_data):
        monitor = self.monitor or ("val_loss" if val_data is not None else "train_loss")
        if monitor.startswith("val_") and val_data is None:
            raise ValueError("Trainer monitor requires val_data for val_* keys")
        mode = self.monitor_mode
        if mode is None:
            mode = "min" if monitor.endswith("_loss") else "max"
        if mode not in {"min", "max"}:
            raise ValueError("monitor_mode must be 'min' or 'max'")
        return monitor, mode

    def _monitor_value(self, monitor, train_summary, val_summary):
        stage, _, key = monitor.partition("_")
        source = {"train": train_summary, "val": val_summary}.get(stage)
        if source is None or key not in source:
            raise ValueError(f"Monitor key {monitor} was not produced by the trainer")
        return source[key]

    def _save_best(self):
        if self.save_best_path is None:
            return
        if self.save_best_path.exists() and self.save_best_path.is_dir():
            raise ValueError("save_best_path must be a file path, not a directory")
        self.save_best_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.best_state_dict, self.save_best_path)

    def evaluate(self, data, stage="val"):
        return self._run_epoch(data, stage=stage, training=False)

    def test(self, data):
        return self.evaluate(data, stage="test")

    def fit(self, train_data, val_data=None):
        monitor, mode = self._resolve_monitor(val_data)
        history = {"epochs": self.max_epochs, "train": [], "val": [], "best_epoch": None, "best_metric": None, "monitor": monitor}
        for epoch in range(1, self.max_epochs + 1):
            train_summary = self._run_epoch(train_data, stage="train", training=True)
            history["train"].append(train_summary)
            val_summary = None
            if val_data is not None:
                val_summary = self._run_epoch(val_data, stage="val", training=False)
                history["val"].append(val_summary)
            current = self._monitor_value(monitor, train_summary, val_summary)
            improved = self.best_metric is None or (current < self.best_metric if mode == "min" else current > self.best_metric)
            if improved:
                self.best_metric = float(current)
                self.best_epoch = epoch
                self.best_state_dict = deepcopy(self.model.state_dict())
                self._save_best()

        history["best_epoch"] = self.best_epoch
        history["best_metric"] = self.best_metric
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        return history
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/train/test_trainer.py tests/train/test_trainer_evaluation.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add vgl/train/trainer.py tests/train/test_trainer.py tests/train/test_trainer_evaluation.py
git commit -m "feat: add training loop evaluation and monitoring"
```

### Task 4: Upgrade the Homogeneous Example and Training Docs

**Files:**
- Modify: `examples/homo/node_classification.py`
- Modify: `tests/integration/test_end_to_end_homo.py`
- Modify: `README.md`
- Modify: `docs/quickstart.md`
- Modify: `docs/core-concepts.md`

**Step 1: Write the failing integration test**

Update `tests/integration/test_end_to_end_homo.py`:

```python
import torch

from examples.homo.node_classification import TinyHomoModel, build_demo_graph
from vgl import Accuracy, NodeClassificationTask, Trainer


def test_end_to_end_homo_training_loop_runs(tmp_path):
    graph = build_demo_graph()
    trainer = Trainer(
        model=TinyHomoModel(),
        task=NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
            metrics=["accuracy"],
        ),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=2,
        monitor="val_accuracy",
        save_best_path=tmp_path / "best.pt",
    )

    history = trainer.fit(graph, val_data=graph)
    test_result = trainer.test(graph)

    assert history["epochs"] == 2
    assert len(history["train"]) == 2
    assert len(history["val"]) == 2
    assert "accuracy" in history["val"][-1]
    assert history["best_epoch"] is not None
    assert "accuracy" in test_result
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integration/test_end_to_end_homo.py -v`
Expected: `FAIL` because the example and homo integration path still expose only the old `fit(graph)` flow.

**Step 3: Write minimal implementation and docs**

Update `examples/homo/node_classification.py` so it exports `build_demo_graph()` and demonstrates the full loop:

```python
from pathlib import Path
import sys
import tempfile

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vgl import Graph, NodeClassificationTask, Trainer


class TinyHomoModel(nn.Module):
    ...


def build_demo_graph():
    return Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
        train_mask=torch.tensor([True, True]),
        val_mask=torch.tensor([True, True]),
        test_mask=torch.tensor([True, True]),
    )


def main():
    graph = build_demo_graph()
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = Trainer(
            model=TinyHomoModel(),
            task=NodeClassificationTask(
                target="y",
                split=("train_mask", "val_mask", "test_mask"),
                metrics=["accuracy"],
            ),
            optimizer=torch.optim.Adam,
            lr=1e-2,
            max_epochs=2,
            monitor="val_accuracy",
            save_best_path=Path(tmp_dir) / "best.pt",
        )
        history = trainer.fit(graph, val_data=graph)
        test_result = trainer.test(graph)
        print({"epochs": history["epochs"], "best_epoch": history["best_epoch"], "test_accuracy": test_result["accuracy"]})
```

Update docs:

- `README.md`: mention epoch-level metrics, `evaluate/test`, and best-checkpoint support
- `docs/quickstart.md`: add a minimal `fit(train, val)` plus `test()` snippet
- `docs/core-concepts.md`: explain the new `Task` / `Metric` / `Trainer` boundary

**Step 4: Run test and example to verify they pass**

Run: `python -m pytest tests/integration/test_end_to_end_homo.py -v`
Expected: `PASS`

Run: `python examples/homo/node_classification.py`
Expected: prints a dictionary containing `epochs`, `best_epoch`, and `test_accuracy`

**Step 5: Commit**

```bash
git add examples/homo/node_classification.py tests/integration/test_end_to_end_homo.py README.md docs/quickstart.md docs/core-concepts.md
git commit -m "feat: add full training loop example"
```

### Task 5: Run Full Regression Verification

**Files:**
- Verify only

**Step 1: Run the focused Phase 5 tests**

Run: `python -m pytest tests/train/test_metrics.py tests/train/test_task_metric_contract.py tests/train/test_trainer.py tests/train/test_trainer_evaluation.py tests/integration/test_end_to_end_homo.py tests/test_package_exports.py -v`
Expected: `PASS`

**Step 2: Run the full test suite**

Run: `python -m pytest -v`
Expected: `PASS`

**Step 3: Run lint and type checking**

Run: `python -m ruff check .`
Expected: `All checks passed`

Run: `python -m mypy vgl`
Expected: `Success: no issues found`

**Step 4: Run the example smoke suite**

Run: `python examples/homo/node_classification.py`
Expected: prints a dictionary containing `epochs`, `best_epoch`, and `test_accuracy`

Run: `python examples/homo/graph_classification.py`
Expected: prints a training result dictionary

Run: `python examples/homo/link_prediction.py`
Expected: prints a training result dictionary

Run: `python examples/hetero/node_classification.py`
Expected: prints a training result dictionary

Run: `python examples/hetero/graph_classification.py`
Expected: prints a training result dictionary

Run: `python examples/temporal/event_prediction.py`
Expected: prints a training result dictionary

**Step 5: Commit final polish only if verification required follow-up edits**

```bash
git add README.md docs/quickstart.md docs/core-concepts.md vgl tests examples
git commit -m "chore: verify training loop flow"
```

Do not create an empty commit. If verification passes without any follow-up edits, stop after recording the evidence.
