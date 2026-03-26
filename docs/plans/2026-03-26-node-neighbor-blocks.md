# Node Neighbor Sampler Block Output Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add optional DGL-style block output to homogeneous `NodeNeighborSampler` while preserving the current sampled subgraph and `NodeBatch` API.

**Architecture:** Record homogeneous per-hop cumulative node frontiers during neighbor expansion, materialize blocks from the sampled subgraph with `to_block(...)`, preserve original graph ids through sampled subgraphs and blocks, and batch block layers independently inside `NodeBatch`.

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

### Task 1: Add failing sampler and batch regression tests

**Files:**
- Modify: `tests/data/test_node_neighbor_sampler.py`
- Modify: `tests/core/test_node_batch.py`
- Modify: `tests/core/test_batch_transfer.py`

**Step 1: Write the failing test**

Add tests proving:

- `NodeNeighborSampler(num_neighbors=[...], output_blocks=True)` returns `SampleRecord.blocks` for homogeneous graphs
- block count equals the configured hop count even when a hop expands no new nodes
- block ordering is outer-to-inner
- `block.src_n_id`, `block.dst_n_id`, and `block.edata["e_id"]` preserve original graph ids from the sampled subgraph
- block edges stay limited to the sampled subgraph rather than the full original graph
- loader materialization exposes `NodeBatch.blocks`
- `NodeBatch.from_samples(...)` batches each block layer across samples
- `NodeBatch.to(...)` and `NodeBatch.pin_memory()` move and pin blocks
- heterogeneous `output_blocks=True` raises a clear `ValueError`

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py tests/core/test_node_batch.py tests/core/test_batch_transfer.py`
Expected: FAIL because sampler records and node batches do not yet carry blocks.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py tests/core/test_node_batch.py tests/core/test_batch_transfer.py`
Expected: FAIL on the missing block-output behavior.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement optional homogeneous block output

**Files:**
- Modify: `vgl/dataloading/sampler.py`
- Modify: `vgl/dataloading/executor.py`
- Modify: `vgl/dataloading/materialize.py`
- Modify: `vgl/dataloading/records.py`
- Modify: `vgl/graph/batch.py`
- Modify: `vgl/ops/khop.py`
- Modify: `vgl/ops/block.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py tests/core/test_node_batch.py tests/core/test_batch_transfer.py`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- `output_blocks` on `NodeNeighborSampler`
- homogeneous-only validation for block output
- homogeneous per-hop cumulative sampling state capture
- block materialization from the sampled subgraph in `materialize_context(...)`
- `SampleRecord.blocks`
- `NodeBatch.blocks` plus per-layer block batching
- `NodeBatch.to(...)` and `NodeBatch.pin_memory()` support for blocks
- `to_block(...)` preservation of pre-existing `n_id` and `e_id`
- clear unsupported errors for heterogeneous or stitched block output requests

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py tests/core/test_node_batch.py tests/core/test_batch_transfer.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Refresh docs and run full regression

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review docs for node-sampling examples that should mention optional block output.

**Step 2: Run test to verify the code path state**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py tests/core/test_node_batch.py tests/core/test_batch_transfer.py`
Expected: PASS

**Step 3: Write minimal implementation**

Document that homogeneous `NodeNeighborSampler` can optionally emit `NodeBatch.blocks` for layer-wise training, and note the current unsupported heterogeneous/distributed limitation.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add node neighbor sampler block output"`
