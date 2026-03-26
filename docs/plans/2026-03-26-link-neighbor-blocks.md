# Link Neighbor Sampler Block Output Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add optional block output to local homogeneous `LinkNeighborSampler` while preserving the existing `LinkPredictionBatch` supervision contract.

**Architecture:** Record homogeneous cumulative frontier states while sampling link neighborhoods, rebuild message-passing blocks from the sampled graph after feature materialization and seed-edge exclusion, and batch those blocks layer-by-layer into `LinkPredictionBatch`.

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

### Task 1: Add failing link block regressions

**Files:**
- Modify: `tests/data/test_link_neighbor_sampler.py`
- Modify: `tests/core/test_link_prediction_batch.py`
- Modify: `tests/core/test_batch_transfer.py`

**Step 1: Write the failing test**

Add tests proving:

- `LinkNeighborSampler(num_neighbors=[...], output_blocks=True)` returns `LinkPredictionRecord.blocks` for local homogeneous graphs
- block order is outer-to-inner and preserves original graph `n_id` / `e_id`
- finite-fanout sampling keeps omitted in-edges out of block edge sets
- `exclude_seed_edges` removes supervision edges from blocks
- loader materialization exposes `LinkPredictionBatch.blocks`
- `LinkPredictionBatch.from_records(...)` batches each block layer across multiple sampled graphs
- `LinkPredictionBatch.to(...)` and `LinkPredictionBatch.pin_memory()` move and pin blocks
- heterogeneous `output_blocks=True` raises a clear `ValueError`

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py tests/core/test_link_prediction_batch.py tests/core/test_batch_transfer.py`
Expected: FAIL because link records, batches, and the sampler do not yet expose blocks.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py tests/core/test_link_prediction_batch.py tests/core/test_batch_transfer.py`
Expected: FAIL on missing block-output behavior.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement local homogeneous link block output

**Files:**
- Modify: `vgl/dataloading/sampler.py`
- Modify: `vgl/dataloading/executor.py`
- Modify: `vgl/dataloading/materialize.py`
- Modify: `vgl/dataloading/records.py`
- Modify: `vgl/graph/batch.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py tests/core/test_link_prediction_batch.py tests/core/test_batch_transfer.py`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- `output_blocks` on `LinkNeighborSampler`
- local homogeneous-only validation for block output
- homogeneous per-hop cumulative frontier tracking for link seeds
- deferred block materialization after feature overlay so blocks inherit fetched tensors
- link-level block construction from the sampled graph after seed-edge exclusion
- `LinkPredictionRecord.blocks`
- `LinkPredictionBatch.blocks` plus per-layer batching
- `LinkPredictionBatch.to(...)` and `LinkPredictionBatch.pin_memory()` support for blocks
- clear unsupported errors for heterogeneous or stitched link block output

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py tests/core/test_link_prediction_batch.py tests/core/test_batch_transfer.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Refresh docs and run full regression

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review docs for link-sampling text that should mention optional block output and the local homogeneous limitation.

**Step 2: Run test to verify the code path state**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py tests/core/test_link_prediction_batch.py tests/core/test_batch_transfer.py`
Expected: PASS

**Step 3: Write minimal implementation**

Document that local homogeneous `LinkNeighborSampler(..., output_blocks=True)` exposes `LinkPredictionBatch.blocks` while preserving the existing supervision graph and endpoint indices.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add link neighbor sampler block output"`
