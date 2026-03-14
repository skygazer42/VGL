# Temporal Event Prediction Phase 3 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a stable temporal event prediction training path for single-event-type temporal graphs using explicit candidate-event samples and the existing generic `Trainer`.

**Architecture:** Keep the current unified `Graph` abstraction intact and add a temporal event sample and batch contract around it. Implement `TemporalEventRecord`, `TemporalEventBatch`, and `TemporalEventPredictionTask`, teach the loader to collate explicit temporal samples, and replace the placeholder temporal example with a real event prediction flow that relies on causal history views. Preserve the existing node classification and graph classification workflows.

**Tech Stack:** Python 3.11+, PyTorch, pytest, ruff, mypy

---

### Task 1: Add Temporal Event Record and Batch Contracts

**Files:**
- Modify: `src/gnn/data/sample.py`
- Modify: `src/gnn/core/batch.py`
- Modify: `src/gnn/core/__init__.py`
- Test: `tests/core/test_temporal_event_batch.py`

**Step 1: Write the failing batch contract test**

```python
import torch

from gnn import Graph
from gnn.core.batch import TemporalEventBatch
from gnn.data.sample import TemporalEventRecord


def test_temporal_event_batch_tracks_fields_and_history_views():
    graph = Graph.temporal(
        nodes={"node": {"x": torch.randn(3, 4)}},
        edges={
            ("node", "interacts", "node"): {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
                "timestamp": torch.tensor([1, 3, 5]),
            }
        },
        time_attr="timestamp",
    )
    records = [
        TemporalEventRecord(graph=graph, src_index=0, dst_index=1, timestamp=3, label=1),
        TemporalEventRecord(graph=graph, src_index=2, dst_index=0, timestamp=5, label=0),
    ]

    batch = TemporalEventBatch.from_records(records)
    history = batch.history_graph(0)
    edge_type = ("node", "interacts", "node")

    assert torch.equal(batch.src_index, torch.tensor([0, 2]))
    assert torch.equal(batch.dst_index, torch.tensor([1, 0]))
    assert torch.equal(batch.timestamp, torch.tensor([3, 5]))
    assert torch.equal(batch.labels, torch.tensor([1, 0]))
    assert torch.equal(history.edges[edge_type].timestamp, torch.tensor([1, 3]))
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/core/test_temporal_event_batch.py -v`
Expected: `FAIL` because `TemporalEventRecord`, `TemporalEventBatch`, or `history_graph()` do not exist yet.

**Step 3: Write minimal implementation**

`src/gnn/data/sample.py`

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TemporalEventRecord:
    graph: Any
    src_index: int
    dst_index: int
    timestamp: int
    label: int
    event_features: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_id: str | None = None
```

Update `src/gnn/core/batch.py` with a new batch type:

```python
@dataclass(slots=True)
class TemporalEventBatch:
    graph: Graph
    src_index: torch.Tensor
    dst_index: torch.Tensor
    timestamp: torch.Tensor
    labels: torch.Tensor
    event_features: torch.Tensor | None = None
    metadata: list[dict] | None = None

    @classmethod
    def from_records(cls, records: list["TemporalEventRecord"]) -> "TemporalEventBatch":
        if not records:
            raise ValueError("TemporalEventBatch requires at least one record")
        graph = records[0].graph
        if graph.schema.time_attr is None:
            raise ValueError("TemporalEventBatch requires a temporal graph with schema.time_attr")
        if any(record.graph is not graph for record in records):
            raise ValueError("TemporalEventBatch currently supports samples from a single source graph only")
        return cls(
            graph=graph,
            src_index=torch.tensor([record.src_index for record in records]),
            dst_index=torch.tensor([record.dst_index for record in records]),
            timestamp=torch.tensor([record.timestamp for record in records]),
            labels=torch.tensor([record.label for record in records]),
            metadata=[record.metadata for record in records],
        )

    def history_graph(self, index: int):
        return self.graph.snapshot(self.timestamp[index].item())
```

Export `TemporalEventBatch` from `src/gnn/core/__init__.py`.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/core/test_temporal_event_batch.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add src/gnn/data/sample.py src/gnn/core/batch.py src/gnn/core/__init__.py tests/core/test_temporal_event_batch.py
git commit -m "feat: add temporal event batch contract"
```

### Task 2: Teach Loader to Collate Temporal Event Records

**Files:**
- Modify: `src/gnn/data/loader.py`
- Modify: `src/gnn/data/__init__.py`
- Test: `tests/data/test_temporal_event_loader.py`

**Step 1: Write the failing loader tests**

```python
import pytest
import torch

from gnn import Graph
from gnn.data.dataset import ListDataset
from gnn.data.loader import Loader
from gnn.data.sample import TemporalEventRecord
from gnn.data.sampler import FullGraphSampler


def _temporal_graph():
    return Graph.temporal(
        nodes={"node": {"x": torch.randn(3, 4)}},
        edges={
            ("node", "interacts", "node"): {
                "edge_index": torch.tensor([[0, 1], [1, 2]]),
                "timestamp": torch.tensor([1, 4]),
            }
        },
        time_attr="timestamp",
    )


def test_loader_collates_temporal_event_records():
    graph = _temporal_graph()
    dataset = ListDataset(
        [
            TemporalEventRecord(graph=graph, src_index=0, dst_index=1, timestamp=1, label=1),
            TemporalEventRecord(graph=graph, src_index=1, dst_index=2, timestamp=4, label=0),
        ]
    )
    loader = Loader(dataset=dataset, sampler=FullGraphSampler(), batch_size=2)

    batch = next(iter(loader))

    assert torch.equal(batch.labels, torch.tensor([1, 0]))
    assert torch.equal(batch.timestamp, torch.tensor([1, 4]))


def test_loader_rejects_temporal_records_from_multiple_graphs():
    dataset = ListDataset(
        [
            TemporalEventRecord(graph=_temporal_graph(), src_index=0, dst_index=1, timestamp=1, label=1),
            TemporalEventRecord(graph=_temporal_graph(), src_index=1, dst_index=2, timestamp=4, label=0),
        ]
    )
    loader = Loader(dataset=dataset, sampler=FullGraphSampler(), batch_size=2)

    with pytest.raises(ValueError, match="single source graph"):
        next(iter(loader))
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_temporal_event_loader.py -v`
Expected: `FAIL` because `Loader` does not yet detect temporal event records.

**Step 3: Write minimal implementation**

Update `src/gnn/data/loader.py`:

```python
from gnn.core.batch import GraphBatch, TemporalEventBatch
from gnn.data.sample import TemporalEventRecord


def _build_batch(self, items):
    if items and isinstance(items[0], TemporalEventRecord):
        return TemporalEventBatch.from_records(items)
    if items and hasattr(items[0], "graph") and self.label_source is not None and self.label_key is not None:
        return GraphBatch.from_samples(
            items,
            label_key=self.label_key,
            label_source=self.label_source,
        )
    return GraphBatch.from_graphs(items)
```

Update `src/gnn/data/__init__.py` so `TemporalEventRecord` is exported next to `SampleRecord`.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/data/test_temporal_event_loader.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add src/gnn/data/loader.py src/gnn/data/__init__.py tests/data/test_temporal_event_loader.py
git commit -m "feat: add temporal event loader collation"
```

### Task 3: Add Temporal Event Prediction Task and Trainer Coverage

**Files:**
- Modify: `src/gnn/train/tasks.py`
- Modify: `src/gnn/train/__init__.py`
- Test: `tests/train/test_temporal_event_task.py`
- Test: `tests/train/test_temporal_event_trainer.py`

**Step 1: Write the failing task and trainer tests**

```python
import pytest
import torch
from torch import nn

from gnn import Graph
from gnn.core.batch import TemporalEventBatch
from gnn.data.sample import TemporalEventRecord
from gnn.train.tasks import TemporalEventPredictionTask
from gnn.train.trainer import Trainer


def _batch():
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
    return TemporalEventBatch.from_records(
        [
            TemporalEventRecord(graph=graph, src_index=0, dst_index=1, timestamp=1, label=1),
            TemporalEventRecord(graph=graph, src_index=1, dst_index=2, timestamp=4, label=0),
        ]
    )


def test_temporal_event_prediction_task_computes_loss():
    task = TemporalEventPredictionTask(target="label")
    logits = torch.randn(2, 2, requires_grad=True)

    loss = task.loss(_batch(), logits, stage="train")

    assert loss.ndim == 0


def test_temporal_event_prediction_task_rejects_unsupported_loss():
    with pytest.raises(ValueError, match="Unsupported loss"):
        TemporalEventPredictionTask(target="label", loss="bce")


class TinyTemporalEventModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(9, 2)

    def forward(self, batch):
        src_x = batch.graph.x[batch.src_index]
        dst_x = batch.graph.x[batch.dst_index]
        history_counts = torch.tensor(
            [batch.history_graph(i).edge_index.size(1) for i in range(batch.labels.size(0))],
            dtype=src_x.dtype,
        ).unsqueeze(-1)
        return self.linear(torch.cat([src_x, dst_x, history_counts], dim=-1))


def test_trainer_runs_temporal_event_prediction_epoch():
    batch = _batch()
    trainer = Trainer(
        model=TinyTemporalEventModel(),
        task=TemporalEventPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit([batch])

    assert history["epochs"] == 1
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/train/test_temporal_event_task.py tests/train/test_temporal_event_trainer.py -v`
Expected: `FAIL` because `TemporalEventPredictionTask` does not exist yet.

**Step 3: Write minimal implementation**

Update `src/gnn/train/tasks.py`:

```python
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
```

Export the task from `src/gnn/train/__init__.py`.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/train/test_temporal_event_task.py tests/train/test_temporal_event_trainer.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add src/gnn/train/tasks.py src/gnn/train/__init__.py tests/train/test_temporal_event_task.py tests/train/test_temporal_event_trainer.py
git commit -m "feat: add temporal event prediction task"
```

### Task 4: Replace the Placeholder Temporal Example and Add Integration Coverage

**Files:**
- Modify: `examples/temporal/event_prediction.py`
- Modify: `tests/integration/test_end_to_end_temporal.py`
- Modify: `README.md`
- Modify: `docs/quickstart.md`
- Modify: `docs/core-concepts.md`

**Step 1: Write the failing integration test**

```python
import torch
from torch import nn

from gnn import Graph
from gnn.data.dataset import ListDataset
from gnn.data.loader import Loader
from gnn.data.sample import TemporalEventRecord
from gnn.data.sampler import FullGraphSampler
from gnn.train.tasks import TemporalEventPredictionTask
from gnn.train.trainer import Trainer


class TinyTemporalEventModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(9, 2)

    def forward(self, batch):
        src_x = batch.graph.x[batch.src_index]
        dst_x = batch.graph.x[batch.dst_index]
        history_counts = torch.tensor(
            [batch.history_graph(i).edge_index.size(1) for i in range(batch.labels.size(0))],
            dtype=src_x.dtype,
        ).unsqueeze(-1)
        return self.linear(torch.cat([src_x, dst_x, history_counts], dim=-1))


def test_end_to_end_temporal_event_prediction_runs():
    graph = Graph.temporal(
        nodes={"node": {"x": torch.randn(3, 4)}},
        edges={
            ("node", "interacts", "node"): {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
                "timestamp": torch.tensor([1, 3, 5]),
            }
        },
        time_attr="timestamp",
    )
    samples = [
        TemporalEventRecord(graph=graph, src_index=0, dst_index=1, timestamp=3, label=1),
        TemporalEventRecord(graph=graph, src_index=2, dst_index=0, timestamp=5, label=0),
    ]
    loader = Loader(dataset=ListDataset(samples), sampler=FullGraphSampler(), batch_size=2)
    trainer = Trainer(
        model=TinyTemporalEventModel(),
        task=TemporalEventPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    result = trainer.fit(loader)

    assert result["epochs"] == 1
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integration/test_end_to_end_temporal.py -v`
Expected: `FAIL` because the example and integration path still use placeholder node classification.

**Step 3: Write minimal implementation**

Update `examples/temporal/event_prediction.py` to mirror the integration flow above and print the `Trainer.fit(...)` result.

Update docs to reflect the new surface:

- `README.md`: mention `TemporalEventPredictionTask` and explicit candidate-event training
- `docs/quickstart.md`: add a temporal event prediction snippet
- `docs/core-concepts.md`: document `TemporalEventRecord` and `TemporalEventBatch`

**Step 4: Run test and example to verify they pass**

Run: `python -m pytest tests/integration/test_end_to_end_temporal.py -v`
Expected: `PASS`

Run: `python examples/temporal/event_prediction.py`
Expected: prints `{'epochs': 1}`

**Step 5: Commit**

```bash
git add examples/temporal/event_prediction.py tests/integration/test_end_to_end_temporal.py README.md docs/quickstart.md docs/core-concepts.md
git commit -m "feat: add temporal event prediction example"
```

### Task 5: Run Full Regression Verification

**Files:**
- Verify only

**Step 1: Run the full test suite**

Run: `python -m pytest -v`
Expected: `PASS`

**Step 2: Run lint**

Run: `python -m ruff check .`
Expected: `All checks passed`

**Step 3: Run type checking**

Run: `python -m mypy src`
Expected: `Success: no issues found`

**Step 4: Re-run the temporal example**

Run: `python examples/temporal/event_prediction.py`
Expected: prints `{'epochs': 1}`

**Step 5: Commit any final polish**

```bash
git add .
git commit -m "chore: verify temporal event prediction flow"
```
