# Distributed Stitched Hetero Temporal Sampling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Teach typed heterogeneous `TemporalNeighborSampler` to stitch cross-partition strict-history structure for shard-local temporal graphs when sampling through a coordinator-backed feature source.

**Architecture:** Keep the public temporal sampler plan unchanged and add one stitched heterogeneous branch inside `PlanExecutor._sample_temporal_neighbors(...)`. That branch will build one filtered relation-local temporal history in global-id space for the sampled `edge_type`, expand typed visited node ids over that filtered history, construct one stitched typed temporal graph with globally aligned node/edge tensors, and rebuild the sampled `TemporalEventRecord` against that graph.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add failing stitched hetero temporal sampling regressions

**Files:**
- Modify: `tests/data/test_temporal_neighbor_sampler.py`
- Modify: `tests/integration/test_foundation_partition_local.py`

**Step 1: Write the failing test**

Add regressions proving:

- a shard-local typed heterogeneous `TemporalEventRecord` sampled through `TemporalNeighborSampler` plus `LocalSamplingCoordinator` can stitch earlier remote typed history for the same `edge_type`
- the stitched typed temporal history keeps per-type global `n_id`, relation-local global `e_id`, timestamps, and optional fetched typed node/edge features aligned through the coordinator
- the public sampled temporal-training loader path can consume the stitched typed temporal batch during temporal event prediction

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_temporal_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched_hetero_temporal`
Expected: FAIL because typed heterogeneous temporal sampling still keeps its history strictly shard-local.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/data/test_temporal_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched_hetero_temporal`
Expected: FAIL on missing remote typed history nodes or edges.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement stitched heterogeneous temporal history expansion and materialization

**Files:**
- Modify: `vgl/dataloading/executor.py`
- Modify: `tests/data/test_temporal_neighbor_sampler.py`
- Modify: `tests/integration/test_foundation_partition_local.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_temporal_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched_hetero_temporal`
Expected: FAIL

**Step 3: Write minimal implementation**

Detect shard-local coordinator-backed typed temporal sampling inside `PlanExecutor._sample_temporal_neighbors(...)`, build a filtered relation-local temporal history in global-id space for the sampled `edge_type`, expand typed visited nodes over that filtered history with the same relation-local fanout semantics as the current hetero temporal sampler, build one stitched typed temporal graph with aligned node and edge tensors, rebuild the sampled `TemporalEventRecord` against that graph, and store typed sampled indices in executor state for later feature overlays.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_temporal_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched_hetero_temporal`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Document stitched hetero temporal sampling scope and run branch verification

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review docs for places that still say heterogeneous temporal stitched sampling is future work.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_temporal_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched_hetero_temporal`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document that shard-local typed heterogeneous `TemporalNeighborSampler` can now stitch cross-partition relation-local history through the local coordinator path.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add distributed stitched hetero temporal sampling"`
