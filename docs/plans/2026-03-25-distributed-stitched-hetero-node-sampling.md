# Distributed Stitched Hetero Node Sampling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Teach heterogeneous `NodeNeighborSampler` to stitch cross-partition frontier structure for shard-local non-temporal hetero graphs when sampling through a coordinator-backed feature source.

**Architecture:** Keep the public node sampler plan unchanged and add one stitched heterogeneous branch inside `PlanExecutor._expand_neighbors(...)`. That branch expands typed visited nodes in global-id space with coordinator incident-edge queries, builds one stitched hetero graph with globally aligned typed features, and hands materialization a prebuilt sample graph plus typed seed positions.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add failing stitched hetero node sampling regressions

**Files:**
- Modify: `tests/data/test_node_neighbor_sampler.py`
- Modify: `tests/integration/test_foundation_partition_local.py`

**Step 1: Write the failing test**

Add regressions proving:

- a shard-local heterogeneous seed sampled through `NodeNeighborSampler` plus `LocalSamplingCoordinator` can cross a partition boundary and include remote typed nodes and relation edges
- the stitched hetero sample keeps typed global `n_id`, typed global `e_id`, and optional fetched typed node/edge features aligned through the coordinator
- the public sampled hetero node-training loader path can consume the stitched shard-local batch during node classification

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched_hetero`
Expected: FAIL because hetero neighbor expansion still only walks shard-local relation-local adjacency.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched_hetero`
Expected: FAIL on missing remote typed stitched nodes or edges.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement stitched heterogeneous node expansion and materialization

**Files:**
- Modify: `vgl/dataloading/executor.py`
- Modify: `tests/data/test_node_neighbor_sampler.py`
- Modify: `tests/integration/test_foundation_partition_local.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched_hetero`
Expected: FAIL

**Step 3: Write minimal implementation**

Detect shard-local heterogeneous coordinator-backed expansion in `PlanExecutor._expand_neighbors(...)`, expand visited typed nodes in global-id space through coordinator typed incident-edge queries with the same per-hop fanout semantics as local hetero node sampling, build one stitched heterogeneous graph with aligned typed node and edge tensors, and hand materialization a prebuilt sample payload plus global-id-aware typed indices for optional feature overlays.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched_hetero`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Document stitched hetero node sampling scope and run branch verification

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review docs for places that still say all heterogeneous sampling stays shard-local.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched_hetero`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document that shard-local non-temporal heterogeneous `NodeNeighborSampler` can now stitch cross-partition frontier structure through the local coordinator path, and state clearly that heterogeneous link and temporal stitched sampling remain future work.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add distributed stitched hetero node sampling"`
