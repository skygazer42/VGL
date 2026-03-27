# Multi-Relation Heterogeneous Sampler Blocks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable heterogeneous node and link samplers to emit full multi-relation `HeteroBlock` layer lists, including local and stitched paths, while keeping homogeneous block output unchanged.

**Architecture:** Reuse the new `to_hetero_block(...)` graph-op primitive as the sampler materialization target for heterogeneous hop snapshots. Widen the batch/record `blocks` contract to `Block | HeteroBlock`, add hetero-block batching support, and switch heterogeneous sampler block materialization from relation-local `to_block(...)` calls to full-layer `to_hetero_block(...)`.

**Tech Stack:** Python 3.11, PyTorch, pytest, VGL graph/sampler/materialization stack

---

### Task 1: Harden `to_hetero_block(...)` for sampler use

**Files:**
- Modify: `vgl/ops/block.py`
- Modify: `tests/ops/test_block_ops.py`

**Step 1: Write the failing tests**

Add tests that verify:
- `to_hetero_block(...)` keeps selected relations present even when one destination frontier is empty
- same-type and cross-type store maps stay stable with empty frontiers

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_block_ops.py -k hetero_block`
Expected: FAIL in the new empty-frontier regression

**Step 3: Write minimal implementation**

Update `vgl/ops/block.py` so `to_hetero_block(...)`:
- initializes source stores for every selected relation source type
- accepts empty per-type frontiers without schema drift
- still preserves public `n_id` / `e_id`

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_block_ops.py -k hetero_block`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/ops/test_block_ops.py vgl/ops/block.py
git commit -m "feat: harden hetero block construction"
```

### Task 2: Widen block batching to support `HeteroBlock`

**Files:**
- Modify: `vgl/graph/batch.py`
- Modify: `tests/core/test_node_batch.py`
- Modify: `tests/core/test_link_prediction_batch.py`
- Modify: `tests/core/test_batch_transfer.py`

**Step 1: Write the failing tests**

Add tests that verify:
- `NodeBatch.from_samples(...)` batches `HeteroBlock` layers
- `LinkPredictionBatch.from_records(...)` batches `HeteroBlock` layers
- `NodeBatch.to(...)` / `LinkPredictionBatch.to(...)` and `pin_memory()` preserve `HeteroBlock`

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/core/test_node_batch.py tests/core/test_link_prediction_batch.py tests/core/test_batch_transfer.py`
Expected: FAIL because block batching currently only understands `Block`

**Step 3: Write minimal implementation**

Update `vgl/graph/batch.py` to:
- widen `blocks` annotations to `Block | HeteroBlock`
- batch heterogeneous block layers through a dedicated `HeteroBlock` path
- fail clearly on mixed `Block` / `HeteroBlock` layers

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/core/test_node_batch.py tests/core/test_link_prediction_batch.py tests/core/test_batch_transfer.py`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/core/test_node_batch.py tests/core/test_link_prediction_batch.py tests/core/test_batch_transfer.py vgl/graph/batch.py
git commit -m "feat: batch hetero sampler blocks"
```

### Task 3: Add failing heterogeneous node sampler regressions

**Files:**
- Modify: `tests/data/test_node_neighbor_sampler.py`
- Modify: `tests/integration/test_foundation_partition_local.py`

**Step 1: Write the failing tests**

Add tests that verify:
- local heterogeneous node sampling with multiple relations now returns `HeteroBlock` layers
- stitched heterogeneous node sampling with multiple relations returns `HeteroBlock` layers
- the old ambiguous multi-inbound node case now succeeds instead of raising

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py -k hetero`
Expected: FAIL because heterogeneous node block output still routes through the relation-local single-inbound path

**Step 3: Write minimal implementation**

Update:
- `vgl/dataloading/sampler.py`
- `vgl/dataloading/executor.py`
- `vgl/dataloading/materialize.py`

so heterogeneous node block materialization uses full per-type hop snapshots and `to_hetero_block(...)`.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k 'hetero and block'`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/data/test_node_neighbor_sampler.py tests/integration/test_foundation_partition_local.py vgl/dataloading/sampler.py vgl/dataloading/executor.py vgl/dataloading/materialize.py
git commit -m "feat: emit hetero blocks for node sampling"
```

### Task 4: Add failing heterogeneous link sampler regressions

**Files:**
- Modify: `tests/data/test_link_neighbor_sampler.py`
- Modify: `tests/integration/test_foundation_partition_local.py`

**Step 1: Write the failing tests**

Add tests that verify:
- local heterogeneous link sampling with mixed `edge_type` supervision now returns `HeteroBlock` layers
- stitched heterogeneous link sampling with mixed `edge_type` supervision now returns `HeteroBlock` layers

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py -k 'hetero and blocks'`
Expected: FAIL because `output_blocks=True` still requires a single `edge_type`

**Step 3: Write minimal implementation**

Update the same sampler/materialization path so heterogeneous link block output:
- stops enforcing one supervised `edge_type` for block output
- builds block layers from the full heterogeneous sampled graph
- preserves seed-edge exclusion in the resulting message-passing layers

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k 'hetero and blocks'`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/data/test_link_neighbor_sampler.py tests/integration/test_foundation_partition_local.py vgl/dataloading/sampler.py vgl/dataloading/executor.py vgl/dataloading/materialize.py
git commit -m "feat: emit hetero blocks for link sampling"
```

### Task 5: Update docs and verify end-to-end

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the doc updates**

Document that:
- homogeneous sampler blocks still use `Block`
- heterogeneous sampler blocks may now be `HeteroBlock`
- heterogeneous node/link block output now supports multi-relation frontiers

**Step 2: Run focused verification**

Run: `python -m pytest -q tests/ops/test_block_ops.py tests/core/test_node_batch.py tests/core/test_link_prediction_batch.py tests/core/test_batch_transfer.py tests/data/test_node_neighbor_sampler.py tests/data/test_link_neighbor_sampler.py`
Expected: PASS

**Step 3: Run full verification**

Run: `python -m pytest -q`
Expected: PASS

**Step 4: Commit**

```bash
git add README.md docs/core-concepts.md docs/quickstart.md
git commit -m "docs: describe hetero sampler blocks"
```

**Step 5: Merge**

```bash
git checkout main
git merge --ff-only hetero-sampler-blocks
git push origin main
git branch -d hetero-sampler-blocks
git worktree remove .worktrees/hetero-sampler-blocks
```
