# VGL Package Rename and Layout Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rename the package and published project from `gnn` to `vgl`, remove the `src/` layout, broaden the root-package API, and make the repository consistently identify as `VGL`.

**Architecture:** Move `src/gnn/` to a top-level `vgl/` package and update all internal absolute imports to `vgl.*`. Expand `vgl.__init__` into a convenience entrypoint for common framework objects, rewrite tests/examples/docs to use `vgl`, and update historical planning documents so the repository presents a single package identity. Preserve current framework behavior and validate the migration with focused package-surface tests plus full regression verification.

**Tech Stack:** Python 3.11+, PyTorch, pytest, ruff, mypy, PowerShell, Hatchling

---

### Task 1: Move the Package to `vgl/` and Define the Root Import Surface

**Files:**
- Modify: `pyproject.toml`
- Move: `src/gnn/` -> `vgl/`
- Modify: `vgl/__init__.py`
- Modify: `vgl/core/__init__.py`
- Modify: `vgl/data/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/train/__init__.py`
- Modify: `vgl/compat/__init__.py`
- Modify: `vgl/compat/dgl.py`
- Modify: `vgl/compat/pyg.py`
- Modify: `vgl/core/batch.py`
- Modify: `vgl/core/graph.py`
- Modify: `vgl/core/schema.py`
- Modify: `vgl/core/view.py`
- Modify: `vgl/data/loader.py`
- Modify: `vgl/data/sampler.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/conv/gat.py`
- Modify: `vgl/nn/conv/gcn.py`
- Modify: `vgl/nn/conv/sage.py`
- Modify: `vgl/train/tasks.py`
- Test: `tests/test_package_exports.py`

**Step 1: Write the failing package-surface test**

```python
from vgl import (
    Graph,
    GraphBatch,
    TemporalEventBatch,
    GraphView,
    GraphSchema,
    ListDataset,
    Loader,
    FullGraphSampler,
    NodeSeedSubgraphSampler,
    SampleRecord,
    TemporalEventRecord,
    MessagePassing,
    Trainer,
    Metric,
    NodeClassificationTask,
    GraphClassificationTask,
    TemporalEventPredictionTask,
    global_mean_pool,
    global_sum_pool,
    global_max_pool,
    __version__,
)


def test_package_exposes_broad_vgl_root_surface():
    assert Graph.__name__ == "Graph"
    assert GraphBatch.__name__ == "GraphBatch"
    assert TemporalEventBatch.__name__ == "TemporalEventBatch"
    assert GraphView.__name__ == "GraphView"
    assert GraphSchema.__name__ == "GraphSchema"
    assert ListDataset.__name__ == "ListDataset"
    assert Loader.__name__ == "Loader"
    assert FullGraphSampler.__name__ == "FullGraphSampler"
    assert NodeSeedSubgraphSampler.__name__ == "NodeSeedSubgraphSampler"
    assert SampleRecord.__name__ == "SampleRecord"
    assert TemporalEventRecord.__name__ == "TemporalEventRecord"
    assert MessagePassing.__name__ == "MessagePassing"
    assert Trainer.__name__ == "Trainer"
    assert Metric.__name__ == "Metric"
    assert NodeClassificationTask.__name__ == "NodeClassificationTask"
    assert GraphClassificationTask.__name__ == "GraphClassificationTask"
    assert TemporalEventPredictionTask.__name__ == "TemporalEventPredictionTask"
    assert callable(global_mean_pool)
    assert callable(global_sum_pool)
    assert callable(global_max_pool)
    assert __version__ == "0.1.0"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_package_exports.py -v`
Expected: `FAIL` with `ModuleNotFoundError: No module named 'vgl'`

**Step 3: Write minimal implementation**

Update `pyproject.toml`:

```toml
[project]
name = "vgl"

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.mypy]
no_site_packages = true
ignore_missing_imports = true
```

Move the package tree in PowerShell:

```powershell
Move-Item -Path src\gnn -Destination vgl
Remove-Item -Path src -Recurse -Force
```

Update `vgl/__init__.py` so it re-exports the broad root surface:

```python
from vgl.core import Graph, GraphBatch, GraphSchema, GraphView, TemporalEventBatch
from vgl.data import (
    FullGraphSampler,
    ListDataset,
    Loader,
    NodeSeedSubgraphSampler,
    SampleRecord,
    TemporalEventRecord,
)
from vgl.nn import MessagePassing, global_max_pool, global_mean_pool, global_sum_pool
from vgl.train import (
    GraphClassificationTask,
    Metric,
    NodeClassificationTask,
    Task,
    TemporalEventPredictionTask,
    Trainer,
)
from vgl.version import __version__

__all__ = [
    "Graph",
    "GraphBatch",
    "TemporalEventBatch",
    "GraphView",
    "GraphSchema",
    "ListDataset",
    "Loader",
    "FullGraphSampler",
    "NodeSeedSubgraphSampler",
    "SampleRecord",
    "TemporalEventRecord",
    "MessagePassing",
    "global_mean_pool",
    "global_sum_pool",
    "global_max_pool",
    "Task",
    "Trainer",
    "Metric",
    "NodeClassificationTask",
    "GraphClassificationTask",
    "TemporalEventPredictionTask",
    "__version__",
]
```

Update all moved package modules that still import `gnn.*` so they now import `vgl.*`.

Update `vgl/train/__init__.py` to export `NodeClassificationTask` in addition to the existing objects.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_package_exports.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add pyproject.toml vgl tests/test_package_exports.py
git commit -m "refactor: rename package to vgl"
```

### Task 2: Rewrite Tests and Examples to Use `vgl` and Remove `src` Layout Assumptions

**Files:**
- Modify: `tests/compat/test_dgl_adapter.py`
- Modify: `tests/compat/test_pyg_adapter.py`
- Modify: `tests/core/test_graph_batch.py`
- Modify: `tests/core/test_graph_batch_graph_classification.py`
- Modify: `tests/core/test_graph_homo.py`
- Modify: `tests/core/test_graph_multi_type.py`
- Modify: `tests/core/test_graph_view.py`
- Modify: `tests/core/test_schema.py`
- Modify: `tests/core/test_temporal_event_batch.py`
- Modify: `tests/data/test_graph_classification_loader.py`
- Modify: `tests/data/test_loader.py`
- Modify: `tests/data/test_subgraph_sampler.py`
- Modify: `tests/data/test_temporal_event_loader.py`
- Modify: `tests/integration/test_end_to_end_hetero.py`
- Modify: `tests/integration/test_end_to_end_homo.py`
- Modify: `tests/integration/test_end_to_end_temporal.py`
- Modify: `tests/integration/test_graph_classification_many_graphs.py`
- Modify: `tests/integration/test_graph_classification_subgraph_samples.py`
- Modify: `tests/nn/test_convs.py`
- Modify: `tests/nn/test_message_passing.py`
- Modify: `tests/nn/test_readout.py`
- Modify: `tests/train/test_graph_classification_task.py`
- Modify: `tests/train/test_graph_classification_trainer.py`
- Modify: `tests/train/test_tasks.py`
- Modify: `tests/train/test_temporal_event_task.py`
- Modify: `tests/train/test_temporal_event_trainer.py`
- Modify: `tests/train/test_trainer.py`
- Modify: `examples/homo/node_classification.py`
- Modify: `examples/homo/graph_classification.py`
- Modify: `examples/hetero/node_classification.py`
- Modify: `examples/hetero/graph_classification.py`
- Modify: `examples/temporal/event_prediction.py`
- Test: `tests/integration/test_end_to_end_temporal.py`

**Step 1: Run a representative integration test to verify the old imports fail**

Run: `python -m pytest tests/integration/test_end_to_end_temporal.py -v`
Expected: `FAIL` with `ModuleNotFoundError: No module named 'gnn'` or import errors caused by stale example imports.

**Step 2: Write minimal implementation**

Rewrite all test imports from `gnn` to `vgl`.

Rewrite all example imports from `gnn` to `vgl`.

Replace current example bootstrap code that points at `src`:

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
```

with a repository-root bootstrap:

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
```

Use the broadened root package where that keeps examples shorter, for example:

```python
from vgl import Graph, Loader, Trainer, GraphClassificationTask
```

Keep subpackage imports where they remain clearer, for example:

```python
from vgl.nn.readout import global_mean_pool
```

**Step 3: Run the representative tests to verify they pass**

Run: `python -m pytest tests/integration/test_end_to_end_temporal.py tests/integration/test_end_to_end_homo.py tests/integration/test_graph_classification_many_graphs.py -v`
Expected: `PASS`

**Step 4: Commit**

```bash
git add tests examples
git commit -m "refactor: migrate tests and examples to vgl"
```

### Task 3: Rewrite Active Docs and Historical Plans to `vgl`

**Files:**
- Modify: `README.md`
- Modify: `docs/quickstart.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/plans/2026-03-14-gnn-framework-design.md`
- Modify: `docs/plans/2026-03-14-gnn-framework.md`
- Modify: `docs/plans/2026-03-15-graph-classification-design.md`
- Modify: `docs/plans/2026-03-15-graph-classification.md`
- Modify: `docs/plans/2026-03-15-temporal-event-prediction-design.md`
- Modify: `docs/plans/2026-03-15-temporal-event-prediction.md`
- Test: `tests/test_repo_identity.py`

**Step 1: Write the failing repository-identity test**

```python
from pathlib import Path


ACTIVE_TARGETS = [
    Path("README.md"),
    Path("docs/quickstart.md"),
    Path("docs/core-concepts.md"),
    Path("docs/plans/2026-03-14-gnn-framework-design.md"),
    Path("docs/plans/2026-03-14-gnn-framework.md"),
    Path("docs/plans/2026-03-15-graph-classification-design.md"),
    Path("docs/plans/2026-03-15-graph-classification.md"),
    Path("docs/plans/2026-03-15-temporal-event-prediction-design.md"),
    Path("docs/plans/2026-03-15-temporal-event-prediction.md"),
]

BANNED_SNIPPETS = [
    'name = "gnn"',
    "from gnn",
    "import gnn",
    "src/gnn",
]


def test_docs_and_config_use_vgl_identity():
    for path in ACTIVE_TARGETS:
        text = path.read_text(encoding="utf-8")
        for banned in BANNED_SNIPPETS:
            assert banned not in text, f"{path} still contains {banned!r}"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_repo_identity.py -v`
Expected: `FAIL` because current docs and historical plan docs still contain `gnn` imports or `src/gnn` paths.

**Step 3: Write minimal implementation**

Rewrite active docs and historical plan contents so they consistently refer to:

- `vgl`
- `from vgl import ...`
- `vgl/...` package paths
- `python -m mypy vgl`

Keep the historical plan filenames unchanged even when their content changes.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_repo_identity.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add README.md docs tests/test_repo_identity.py
git commit -m "docs: align repository docs with vgl"
```

### Task 4: Remove Remaining `gnn` / `src/gnn` References and Verify Repository Consistency

**Files:**
- Verify and patch any remaining references in `pyproject.toml`, `vgl/`, `tests/`, `examples/`, `README.md`, and `docs/`

**Step 1: Run consistency searches**

Run: `rg -n "from gnn|import gnn|src/gnn|name = \"gnn\"" pyproject.toml README.md docs examples tests vgl`
Expected: matches remain and identify any files missed in earlier tasks.

**Step 2: Write minimal implementation**

Patch any remaining matches so the repository has one active identity:

- `vgl`
- top-level `vgl/`
- no `src/gnn`

Do not add compatibility aliases; delete the old assumptions instead.

**Step 3: Re-run consistency searches**

Run: `rg -n "from gnn|import gnn|src/gnn|name = \"gnn\"" pyproject.toml README.md docs examples tests vgl`
Expected: no output

**Step 4: Commit**

```bash
git add pyproject.toml README.md docs examples tests vgl
git commit -m "chore: remove remaining gnn references"
```

### Task 5: Run Full Regression Verification on the Renamed Package

**Files:**
- Verify only

**Step 1: Run the full test suite**

Run: `python -m pytest -v`
Expected: `PASS`

**Step 2: Run lint**

Run: `python -m ruff check .`
Expected: `All checks passed`

**Step 3: Run type checking against the renamed package**

Run: `python -m mypy vgl`
Expected: `Success: no issues found`

**Step 4: Run the key examples**

Run: `python examples/homo/node_classification.py`
Expected: prints `{'epochs': 1}`

Run: `python examples/homo/graph_classification.py`
Expected: prints `{'epochs': 1}`

Run: `python examples/hetero/node_classification.py`
Expected: prints `{'epochs': 1}`

Run: `python examples/hetero/graph_classification.py`
Expected: prints `{'epochs': 1}`

Run: `python examples/temporal/event_prediction.py`
Expected: prints `{'epochs': 1}`

**Step 5: Commit any final polish**

```bash
git add .
git commit -m "chore: verify vgl package migration"
```
