# Distributed Stitched Link Sampling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Teach homogeneous `LinkNeighborSampler` to stitch cross-partition frontier structure for shard-local graphs when sampling through a coordinator-backed feature source.

**Architecture:** Keep the public link sampler plan unchanged and extend `PlanExecutor._sample_link_neighbors(...)` with one stitched homogeneous branch. That branch reuses the distributed stitched homogeneous node frontier helpers, builds one in-memory stitched graph, remaps link record endpoints into it, and lets existing feature materialization continue to align through global `n_id` / `e_id`.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add failing stitched link sampling regressions

**Files:**
- Modify: `tests/data/test_link_neighbor_sampler.py`
- Modify: `tests/integration/test_foundation_partition_local.py`

**Step 1: Write the failing test**

Add regressions proving:

- a shard-local homogeneous link record sampled through `LinkNeighborSampler` plus `LocalSamplingCoordinator` can cross a partition boundary and include remote nodes and edges
- the stitched link sample keeps globally aligned `n_id`, `e_id`, and optional fetched node features
- the public sampled link-training loader path can consume the stitched shard-local batch during link prediction training

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched_link`
Expected: FAIL because `sample_link_neighbors` still samples only from shard-local adjacency.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched_link`
Expected: FAIL on missing remote stitched nodes or edges.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement stitched homogeneous link sampling

**Files:**
- Modify: `vgl/dataloading/executor.py`
- Modify: `tests/data/test_link_neighbor_sampler.py`
- Modify: `tests/integration/test_foundation_partition_local.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched_link`
Expected: FAIL

**Step 3: Write minimal implementation**

Detect shard-local homogeneous coordinator-backed link sampling in `PlanExecutor._sample_link_neighbors(...)`, expand the union of link endpoints in global-id space through coordinator incident-edge queries, build one stitched homogeneous graph with globally aligned tensors, remap each link record's endpoints into the stitched graph, and store sampled graph indices so later feature-fetch stages keep working.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched_link`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Document stitched link sampling scope and run branch verification

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review docs for places that still say stitched sampling is node-only.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched_link`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document that shard-local homogeneous `LinkNeighborSampler` can now stitch cross-partition frontier structure through the local coordinator path, and state clearly that heterogeneous and temporal stitched sampling remain future work.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add distributed stitched link sampling"`
