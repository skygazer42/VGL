# Distributed Stitched Node Block Output Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add optional DGL-style block output to stitched homogeneous `NodeNeighborSampler` batches produced through `LocalSamplingCoordinator`.

**Architecture:** Extend the stitched homogeneous node-expansion branch to retain cumulative hop frontiers in global-id space, then rebuild blocks from the stitched sampled graph during materialization after feature overlays have been applied. Keep the public sampler, loader, sample, and batch contracts unchanged.

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

### Task 1: Add failing stitched-node block regressions

**Files:**
- Modify: `tests/data/test_node_neighbor_sampler.py`
- Modify: `tests/integration/test_foundation_partition_local.py`

**Step 1: Write the failing test**

Add regressions proving:

- `Loader(..., sampler=NodeNeighborSampler(..., output_blocks=True), feature_store=coordinator)` materializes `NodeBatch.blocks` for stitched homogeneous shard-local sampling
- stitched blocks preserve remote frontier global `n_id` / `e_id`
- stitched block count matches `len(num_neighbors)` even when a later hop adds no new nodes
- coordinator-backed fetched node/edge tensors remain visible on stitched block graphs after materialization

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py -k "stitched and output_blocks" tests/integration/test_foundation_partition_local.py -k blocks`
Expected: FAIL because the stitched homogeneous executor path still raises an unsupported error for `output_blocks=True` and the materialization path does not rebuild stitched node blocks.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py -k "stitched and output_blocks" tests/integration/test_foundation_partition_local.py -k blocks`
Expected: FAIL on the stitched homogeneous unsupported error or missing block materialization.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement stitched homogeneous node block output

**Files:**
- Modify: `vgl/dataloading/executor.py`
- Modify: `vgl/dataloading/materialize.py`
- Modify: `tests/data/test_node_neighbor_sampler.py`
- Modify: `tests/integration/test_foundation_partition_local.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py -k "stitched and output_blocks" tests/integration/test_foundation_partition_local.py -k blocks`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- stitched homogeneous hop capture in global-id space when `output_blocks=True`
- executor state for stitched node block frontiers without disturbing existing local homogeneous behavior
- materialization-time block rebuilding for `SampleRecord` payloads that were prebuilt by the stitched executor branch
- stitched block construction from the stitched sampled graph using `graph.n_id` to map global hop ids back to sampled local positions
- preservation of fetched node/edge overlays on stitched block graphs
- the current explicit unsupported errors for heterogeneous stitched block output

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py -k "stitched and output_blocks" tests/integration/test_foundation_partition_local.py -k blocks`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Refresh docs and run focused block/coordinator regression

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review docs for places that still say node block output is limited to local homogeneous sampling.

**Step 2: Run test to verify the code path state**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py tests/core/test_node_batch.py tests/core/test_batch_transfer.py tests/integration/test_foundation_partition_local.py`
Expected: PASS if the stitched node block path is implemented cleanly.

**Step 3: Write minimal implementation**

Document that `NodeNeighborSampler(..., output_blocks=True)` now supports stitched homogeneous shard-local sampling through a coordinator-backed feature source, while heterogeneous stitched block output remains unsupported.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add stitched node sampler block output"`
