# Typed Temporal Events Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the temporal event pipeline so typed heterogeneous temporal graphs can flow through `TemporalEventRecord`, `TemporalEventBatch`, `TemporalNeighborSampler`, `Loader`, and `Trainer` without falling back to homogeneous-only assumptions.

**Architecture:** Preserve the current homogeneous temporal path, mirror the typed batch contract already used by `LinkPredictionBatch`, and keep strict-history temporal sampling scoped to the selected relation for typed heterogeneous graphs.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add failing typed temporal regressions

**Files:**
- Modify: `tests/core/test_temporal_event_batch.py`
- Modify: `tests/core/test_batch_transfer.py`
- Modify: `tests/data/test_temporal_neighbor_sampler.py`
- Modify: `tests/data/test_temporal_event_loader.py`
- Modify: `tests/train/test_temporal_event_trainer.py`

**Step 1: Write the failing test**

Add regressions proving:

- `TemporalEventRecord` accepts `edge_type`
- `TemporalEventBatch` batches typed heterogeneous temporal records and preserves typed metadata
- `TemporalNeighborSampler` extracts relation-local hetero history subgraphs
- typed temporal batches survive `to(...)`, `pin_memory()`, loader collation, and one trainer epoch

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/core/test_temporal_event_batch.py tests/core/test_batch_transfer.py tests/data/test_temporal_neighbor_sampler.py tests/data/test_temporal_event_loader.py tests/train/test_temporal_event_trainer.py`
Expected: FAIL because the temporal path still assumes homogeneous graphs only.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/core/test_temporal_event_batch.py tests/core/test_batch_transfer.py tests/data/test_temporal_neighbor_sampler.py tests/data/test_temporal_event_loader.py tests/train/test_temporal_event_trainer.py`
Expected: FAIL on missing typed temporal batch or sampler support.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement typed temporal batching and relation-local sampling

**Files:**
- Modify: `vgl/dataloading/records.py`
- Modify: `vgl/dataloading/sampler.py`
- Modify: `vgl/graph/batch.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/core/test_temporal_event_batch.py tests/core/test_batch_transfer.py tests/data/test_temporal_neighbor_sampler.py tests/data/test_temporal_event_loader.py tests/train/test_temporal_event_trainer.py`
Expected: FAIL

**Step 3: Write minimal implementation**

Teach `TemporalEventRecord` to carry `edge_type`, extend `TemporalEventBatch` with typed metadata mirroring `LinkPredictionBatch`, preserve single-relation temporal edge types and `time_attr` during multi-graph batching, and make `TemporalNeighborSampler` build relation-local strict-history subgraphs and typed feature-fetch stages for heterogeneous temporal graphs.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/core/test_temporal_event_batch.py tests/core/test_batch_transfer.py tests/data/test_temporal_neighbor_sampler.py tests/data/test_temporal_event_loader.py tests/train/test_temporal_event_trainer.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Refresh docs and run branch verification

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`
- Create: `docs/plans/2026-03-25-typed-temporal-events.md`
- Create: `docs/plans/2026-03-25-typed-temporal-events-design.md`

**Step 1: Write the failing test**

No code test. Review docs for places that still describe temporal event prediction as homogeneous-only or omit the typed contract.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document `edge_type` on temporal records, typed temporal batch fields, and relation-local temporal neighbor sampling for heterogeneous graphs.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: support typed temporal event batches"`
