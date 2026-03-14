# Link Prediction Phase 4 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a stable homogeneous link prediction training path for explicit candidate edges using the existing generic `Trainer`.

**Architecture:** Keep the current unified `Graph` abstraction intact and add an explicit link prediction sample and batch contract around it. Implement `LinkPredictionRecord`, `LinkPredictionBatch`, and `LinkPredictionTask`, teach the loader to collate explicit link samples, and add a real homogeneous example plus integration coverage. Preserve the existing node classification, graph classification, and temporal event prediction workflows.

**Tech Stack:** Python 3.11+, PyTorch, pytest, ruff, mypy

**Execution Rules:** Use `@test-driven-development` for every code task, keep commits small, and use `@verification-before-completion` before claiming the phase is done.

---

### Task 1: Add Link Prediction Record and Batch Contracts

**Files:**
- Modify: `vgl/data/sample.py`
- Modify: `vgl/core/batch.py`
- Modify: `vgl/core/__init__.py`
- Modify: `vgl/data/__init__.py`
- Test: `tests/core/test_link_prediction_batch.py`

**Step 1: Write the failing batch contract tests**

```python
import pytest
import torch

from vgl import Graph
from vgl.core.batch import LinkPredictionBatch
from vgl.data.sample import LinkPredictionRecord


def _graph():
    return Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 4),
    )


def test_link_prediction_batch_tracks_fields():
    graph = _graph()
    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1, sample_id="p"),
            LinkPredictionRecord(graph=graph, src_index=2, dst_index=0, label=0, sample_id="n"),
        ]
    )

    assert batch.graph is graph
    assert torch.equal(batch.src_index, torch.tensor([0, 2]))
    assert torch.equal(batch.dst_index, torch.tensor([1, 0]))
    assert torch.equal(batch.labels, torch.tensor([1.0, 0.0]))
    assert batch.metadata == [{}, {}]


def test_link_prediction_batch_rejects_empty_records():
    with pytest.raises(ValueError, match="at least one record"):
        LinkPredictionBatch.from_records([])


def test_link_prediction_batch_rejects_mixed_graphs():
    with pytest.raises(ValueError, match="single source graph"):
        LinkPredictionBatch.from_records(
            [
                LinkPredictionRecord(graph=_graph(), src_index=0, dst_index=1, label=1),
                LinkPredictionRecord(graph=_graph(), src_index=1, dst_index=2, label=0),
            ]
        )


def test_link_prediction_batch_rejects_out_of_range_indices():
    graph = _graph()

    with pytest.raises(ValueError, match="node range"):
        LinkPredictionBatch.from_records(
            [LinkPredictionRecord(graph=graph, src_index=3, dst_index=1, label=1)]
        )


def test_link_prediction_batch_rejects_non_binary_labels():
    graph = _graph()

    with pytest.raises(ValueError, match="binary 0/1"):
        LinkPredictionBatch.from_records(
            [LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=2)]
        )
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/core/test_link_prediction_batch.py -v`
Expected: `FAIL` because `LinkPredictionRecord` and `LinkPredictionBatch` do not exist yet.

**Step 3: Write minimal implementation**

Add a new record to `vgl/data/sample.py`:

```python
@dataclass(slots=True)
class LinkPredictionRecord:
    graph: Any
    src_index: int
    dst_index: int
    label: int
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_id: str | None = None
```

Add a new batch to `vgl/core/batch.py`:

```python
@dataclass(slots=True)
class LinkPredictionBatch:
    graph: Graph
    src_index: torch.Tensor
    dst_index: torch.Tensor
    labels: torch.Tensor
    metadata: list[dict] | None = None

    @classmethod
    def from_records(cls, records: list["LinkPredictionRecord"]) -> "LinkPredictionBatch":
        if not records:
            raise ValueError("LinkPredictionBatch requires at least one record")
        graph = records[0].graph
        if any(record.graph is not graph for record in records):
            raise ValueError("LinkPredictionBatch currently supports samples from a single source graph only")

        src_index = torch.tensor([record.src_index for record in records], dtype=torch.long)
        dst_index = torch.tensor([record.dst_index for record in records], dtype=torch.long)
        labels = torch.tensor([float(record.label) for record in records], dtype=torch.float32)
        num_nodes = graph.x.size(0)

        if ((src_index < 0) | (src_index >= num_nodes) | (dst_index < 0) | (dst_index >= num_nodes)).any():
            raise ValueError("LinkPredictionBatch indices must fall within the source graph node range")
        if not torch.all((labels == 0) | (labels == 1)):
            raise ValueError("LinkPredictionBatch labels must be binary 0/1")

        return cls(
            graph=graph,
            src_index=src_index,
            dst_index=dst_index,
            labels=labels,
            metadata=[record.metadata for record in records],
        )
```

Export both new types from `vgl/core/__init__.py` and `vgl/data/__init__.py`.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/core/test_link_prediction_batch.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add vgl/data/sample.py vgl/core/batch.py vgl/core/__init__.py vgl/data/__init__.py tests/core/test_link_prediction_batch.py
git commit -m "feat: add link prediction batch contract"
```

### Task 2: Teach Loader to Collate Link Prediction Records

**Files:**
- Modify: `vgl/data/loader.py`
- Test: `tests/data/test_link_prediction_loader.py`

**Step 1: Write the failing loader tests**

```python
import pytest
import torch

from vgl import Graph
from vgl.data.dataset import ListDataset
from vgl.data.loader import Loader
from vgl.data.sample import LinkPredictionRecord
from vgl.data.sampler import FullGraphSampler


def _graph():
    return Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 4),
    )


def test_loader_collates_link_prediction_records():
    graph = _graph()
    dataset = ListDataset(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1),
            LinkPredictionRecord(graph=graph, src_index=2, dst_index=0, label=0),
        ]
    )
    loader = Loader(dataset=dataset, sampler=FullGraphSampler(), batch_size=2)

    batch = next(iter(loader))

    assert batch.graph is graph
    assert torch.equal(batch.labels, torch.tensor([1.0, 0.0]))


def test_loader_rejects_link_prediction_records_from_multiple_graphs():
    dataset = ListDataset(
        [
            LinkPredictionRecord(graph=_graph(), src_index=0, dst_index=1, label=1),
            LinkPredictionRecord(graph=_graph(), src_index=1, dst_index=2, label=0),
        ]
    )
    loader = Loader(dataset=dataset, sampler=FullGraphSampler(), batch_size=2)

    with pytest.raises(ValueError, match="single source graph"):
        next(iter(loader))
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_link_prediction_loader.py -v`
Expected: `FAIL` because `Loader` does not yet detect `LinkPredictionRecord`.

**Step 3: Write minimal implementation**

Update `vgl/data/loader.py`:

```python
from vgl.core.batch import GraphBatch, LinkPredictionBatch, TemporalEventBatch
from vgl.data.sample import LinkPredictionRecord, TemporalEventRecord


def _build_batch(self, items):
    if items and isinstance(items[0], TemporalEventRecord):
        return TemporalEventBatch.from_records(items)
    if items and isinstance(items[0], LinkPredictionRecord):
        return LinkPredictionBatch.from_records(items)
    if items and hasattr(items[0], "graph") and self.label_source is not None and self.label_key is not None:
        return GraphBatch.from_samples(
            items,
            label_key=self.label_key,
            label_source=self.label_source,
        )
    return GraphBatch.from_graphs(items)
```

Keep the link prediction dispatch explicit and ahead of the generic `hasattr(item, "graph")` path.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/data/test_link_prediction_loader.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add vgl/data/loader.py tests/data/test_link_prediction_loader.py
git commit -m "feat: add link prediction loader collation"
```

### Task 3: Add Link Prediction Task and Public Exports

**Files:**
- Modify: `vgl/train/tasks.py`
- Modify: `vgl/train/__init__.py`
- Modify: `vgl/__init__.py`
- Test: `tests/train/test_link_prediction_task.py`
- Modify: `tests/test_package_exports.py`

**Step 1: Write the failing task and export tests**

```python
import pytest
import torch

from vgl import Graph
from vgl.core.batch import LinkPredictionBatch
from vgl.data.sample import LinkPredictionRecord
from vgl.train.tasks import LinkPredictionTask


def _batch():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 4),
    )
    return LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1),
            LinkPredictionRecord(graph=graph, src_index=2, dst_index=0, label=0),
        ]
    )


def test_link_prediction_task_computes_bce_loss():
    task = LinkPredictionTask(target="label")
    logits = torch.randn(2, requires_grad=True)

    loss = task.loss(_batch(), logits, stage="train")

    assert loss.ndim == 0


def test_link_prediction_task_rejects_unsupported_loss():
    with pytest.raises(ValueError, match="Unsupported loss"):
        LinkPredictionTask(target="label", loss="cross_entropy")


def test_link_prediction_task_rejects_shape_mismatch():
    task = LinkPredictionTask(target="label")
    logits = torch.randn(2, 2, requires_grad=True)

    with pytest.raises(ValueError, match="one logit per candidate edge"):
        task.loss(_batch(), logits, stage="train")
```

Update `tests/test_package_exports.py` so the root package also exposes:

```python
from vgl import LinkPredictionBatch, LinkPredictionRecord, LinkPredictionTask

assert LinkPredictionBatch.__name__ == "LinkPredictionBatch"
assert LinkPredictionRecord.__name__ == "LinkPredictionRecord"
assert LinkPredictionTask.__name__ == "LinkPredictionTask"
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/train/test_link_prediction_task.py tests/test_package_exports.py -v`
Expected: `FAIL` because `LinkPredictionTask` and root exports do not exist yet.

**Step 3: Write minimal implementation**

Add a new task to `vgl/train/tasks.py`:

```python
class LinkPredictionTask(Task):
    def __init__(self, target="label", loss="binary_cross_entropy", metrics=None):
        if loss != "binary_cross_entropy":
            raise ValueError(f"Unsupported loss: {loss}")
        self.target = target
        self.loss_name = loss
        self.metrics = metrics or []

    def loss(self, batch, logits, stage):
        del stage
        if logits.ndim == 2 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        if logits.ndim != 1 or logits.size(0) != batch.labels.size(0):
            raise ValueError("LinkPredictionTask expects one logit per candidate edge")
        targets = batch.labels.to(dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets)
```

Export `LinkPredictionTask` from `vgl/train/__init__.py` and export `LinkPredictionBatch`, `LinkPredictionRecord`, and `LinkPredictionTask` from `vgl/__init__.py`.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/train/test_link_prediction_task.py tests/test_package_exports.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add vgl/train/tasks.py vgl/train/__init__.py vgl/__init__.py tests/train/test_link_prediction_task.py tests/test_package_exports.py
git commit -m "feat: add link prediction task"
```

### Task 4: Add Trainer Coverage, Example, Integration Test, and Docs

**Files:**
- Create: `tests/train/test_link_prediction_trainer.py`
- Create: `tests/integration/test_end_to_end_link_prediction.py`
- Create: `examples/homo/link_prediction.py`
- Modify: `README.md`
- Modify: `docs/quickstart.md`
- Modify: `docs/core-concepts.md`

**Step 1: Write the failing trainer and integration tests**

```python
import torch
from torch import nn

from vgl import Graph
from vgl.core.batch import LinkPredictionBatch
from vgl.data.sample import LinkPredictionRecord
from vgl.train.tasks import LinkPredictionTask
from vgl.train.trainer import Trainer


def _batch():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 4),
    )
    return LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1),
            LinkPredictionRecord(graph=graph, src_index=2, dst_index=0, label=0),
        ]
    )


class TinyLinkPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.scorer = nn.Linear(8, 1)

    def forward(self, batch):
        node_repr = self.encoder(batch.graph.x)
        src_x = node_repr[batch.src_index]
        dst_x = node_repr[batch.dst_index]
        return self.scorer(torch.cat([src_x, dst_x], dim=-1)).squeeze(-1)


def test_trainer_runs_link_prediction_epoch():
    trainer = Trainer(
        model=TinyLinkPredictor(),
        task=LinkPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit([_batch()])

    assert history["epochs"] == 1
```

Create an integration test that imports the example surface:

```python
import torch

from examples.homo.link_prediction import TinyLinkPredictor, build_demo_loader
from vgl.train.tasks import LinkPredictionTask
from vgl.train.trainer import Trainer


def test_end_to_end_link_prediction_runs():
    trainer = Trainer(
        model=TinyLinkPredictor(),
        task=LinkPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    result = trainer.fit(build_demo_loader())

    assert result["epochs"] == 1
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/train/test_link_prediction_trainer.py tests/integration/test_end_to_end_link_prediction.py -v`
Expected: `FAIL` because the example file does not exist yet.

**Step 3: Write minimal implementation and docs**

Create `examples/homo/link_prediction.py`:

```python
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vgl import Graph
from vgl.data.dataset import ListDataset
from vgl.data.loader import Loader
from vgl.data.sample import LinkPredictionRecord
from vgl.data.sampler import FullGraphSampler
from vgl.train.tasks import LinkPredictionTask
from vgl.train.trainer import Trainer


class TinyLinkPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.scorer = nn.Linear(8, 1)

    def forward(self, batch):
        node_repr = self.encoder(batch.graph.x)
        src_x = node_repr[batch.src_index]
        dst_x = node_repr[batch.dst_index]
        return self.scorer(torch.cat([src_x, dst_x], dim=-1)).squeeze(-1)


def build_demo_loader():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 4),
    )
    samples = [
        LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1),
        LinkPredictionRecord(graph=graph, src_index=2, dst_index=0, label=0),
    ]
    return Loader(
        dataset=ListDataset(samples),
        sampler=FullGraphSampler(),
        batch_size=2,
    )


def main():
    trainer = Trainer(
        model=TinyLinkPredictor(),
        task=LinkPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )
    result = trainer.fit(build_demo_loader())
    print(result)
    return result


if __name__ == "__main__":
    main()
```

Update docs:

- `README.md`: add `LinkPredictionTask` to the supported training surface and add `python examples/homo/link_prediction.py` to the examples list
- `docs/quickstart.md`: add a short explicit-candidate-edge snippet for homogeneous link prediction
- `docs/core-concepts.md`: add `LinkPredictionRecord` and `LinkPredictionBatch`, and list link prediction in the supported training modes

**Step 4: Run tests and example to verify they pass**

Run: `python -m pytest tests/train/test_link_prediction_trainer.py tests/integration/test_end_to_end_link_prediction.py -v`
Expected: `PASS`

Run: `python examples/homo/link_prediction.py`
Expected: prints `{'epochs': 1}`

**Step 5: Commit**

```bash
git add tests/train/test_link_prediction_trainer.py tests/integration/test_end_to_end_link_prediction.py examples/homo/link_prediction.py README.md docs/quickstart.md docs/core-concepts.md
git commit -m "feat: add homogeneous link prediction flow"
```

### Task 5: Run Full Regression Verification

**Files:**
- Verify only

**Step 1: Run the new focused link prediction tests**

Run: `python -m pytest tests/core/test_link_prediction_batch.py tests/data/test_link_prediction_loader.py tests/train/test_link_prediction_task.py tests/train/test_link_prediction_trainer.py tests/integration/test_end_to_end_link_prediction.py tests/test_package_exports.py -v`
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
Expected: prints `{'epochs': 1}`

Run: `python examples/homo/graph_classification.py`
Expected: prints `{'epochs': 1}`

Run: `python examples/homo/link_prediction.py`
Expected: prints `{'epochs': 1}`

Run: `python examples/hetero/node_classification.py`
Expected: prints `{'epochs': 1}`

Run: `python examples/hetero/graph_classification.py`
Expected: prints `{'epochs': 1}`

Run: `python examples/temporal/event_prediction.py`
Expected: prints `{'epochs': 1}`

**Step 5: Commit final polish only if verification required follow-up edits**

```bash
git add README.md docs/quickstart.md docs/core-concepts.md vgl tests examples
git commit -m "chore: verify link prediction flow"
```

Do not create an empty commit. If verification passes without any follow-up edits, stop after recording the evidence.
