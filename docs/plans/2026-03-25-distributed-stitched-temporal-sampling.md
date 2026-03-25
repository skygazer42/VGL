# Distributed Stitched Temporal Sampling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Teach homogeneous `TemporalNeighborSampler` to stitch cross-partition history for shard-local temporal graphs when sampling through a coordinator-backed feature source.

**Architecture:** Keep the public temporal sampler plan unchanged and extend `PlanExecutor._sample_temporal_neighbors(...)` with one stitched homogeneous temporal branch. That branch builds a global filtered history edge set in coordinator space, expands the temporal frontier over that filtered history, constructs one stitched temporal graph with globally aligned `n_id` / `e_id`, and remaps the sampled `TemporalEventRecord` into it so later feature materialization keeps working.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add failing stitched temporal sampling regressions

**Files:**
- Modify: `tests/data/test_temporal_neighbor_sampler.py`
- Modify: `tests/integration/test_foundation_partition_local.py`

**Step 1: Write the failing test**

Add regressions proving:

- a shard-local homogeneous temporal event sampled through `TemporalNeighborSampler` plus `LocalSamplingCoordinator` can cross a partition boundary and include earlier remote history nodes and edges
- the stitched temporal sample keeps globally aligned `n_id`, `e_id`, `timestamp`, and optional fetched node/edge features
- the public sampled temporal-training loader path can consume the stitched shard-local history batch during temporal event prediction training

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_temporal_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched_temporal`
Expected: FAIL because `sample_temporal_neighbors` still samples only from shard-local temporal history.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/data/test_temporal_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched_temporal`
Expected: FAIL on missing remote stitched history nodes or edges.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement stitched homogeneous temporal sampling

**Files:**
- Modify: `vgl/dataloading/executor.py`
- Modify: `tests/data/test_temporal_neighbor_sampler.py`
- Modify: `tests/integration/test_foundation_partition_local.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_temporal_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched_temporal`
Expected: FAIL

**Step 3: Write minimal implementation**

Detect shard-local homogeneous coordinator-backed temporal sampling in `PlanExecutor._sample_temporal_neighbors(...)`, build a global eligible history edge set with the same `strict_history` / `time_window` / `max_events` semantics as local temporal sampling, expand the frontier over that filtered history, build one stitched homogeneous temporal graph with globally aligned tensors, remap the temporal record into the stitched graph, and store sampled graph indices so later feature-fetch stages keep working.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_temporal_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched_temporal`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Document stitched temporal sampling scope and run branch verification

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review docs for places that still say temporal stitched sampling is unsupported.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_temporal_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched_temporal`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document that shard-local homogeneous `TemporalNeighborSampler` can now stitch earlier cross-partition history through the local coordinator path, and state clearly that heterogeneous stitched sampling remains future work.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add distributed stitched temporal sampling"`
