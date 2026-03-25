# Distributed Stitched Node Sampling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Teach homogeneous `NodeNeighborSampler` to stitch cross-partition frontier structure for shard-local graphs when sampling through a coordinator-backed feature source.

**Architecture:** Keep the public sampler plan unchanged and add one stitched homogeneous branch inside `PlanExecutor._expand_neighbors(...)`. That branch expands in global-id space with coordinator incident-edge queries, builds an in-memory stitched graph with globally aligned features, and hands materialization a prebuilt sample graph plus seed positions.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add failing stitched node sampling regressions

**Files:**
- Modify: `tests/data/test_node_neighbor_sampler.py`
- Modify: `tests/integration/test_foundation_partition_local.py`

**Step 1: Write the failing test**

Add regressions proving:

- a shard-local homogeneous seed sampled through `NodeNeighborSampler` plus `LocalSamplingCoordinator` can cross a partition boundary and include remote nodes and edges
- the stitched sample keeps global `n_id`, global `e_id`, and feature-aligned `x`
- the public sampled training loader path can consume the stitched shard-local batch during node classification

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched`
Expected: FAIL because neighbor expansion still only walks shard-local adjacency and the stitched remote frontier is missing.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched`
Expected: FAIL on missing remote stitched nodes or edges.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement stitched homogeneous expansion and materialization

**Files:**
- Modify: `vgl/dataloading/executor.py`
- Modify: `vgl/dataloading/materialize.py`
- Modify: `tests/data/test_node_neighbor_sampler.py`
- Modify: `tests/integration/test_foundation_partition_local.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched`
Expected: FAIL

**Step 3: Write minimal implementation**

Detect shard-local homogeneous coordinator-backed expansion in `PlanExecutor._expand_neighbors(...)`, expand visited nodes in global-id space through coordinator incident-edge queries, build an in-memory stitched graph with aligned node and edge tensors, and hand materialization a prebuilt sample payload plus global-id-aware indices for optional feature overlays.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Document stitched sampling scope and run branch verification

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review docs for places that imply shard-local node sampling cannot cross partition boundaries.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document that VGL now supports stitched homogeneous node sampling for shard-local graphs through the local coordinator path, and state clearly that hetero, link, and temporal stitched sampling are still future work.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add distributed stitched node sampling"`
