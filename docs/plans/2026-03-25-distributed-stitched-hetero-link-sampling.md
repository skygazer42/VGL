# Distributed Stitched Hetero Link Sampling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Teach heterogeneous `LinkNeighborSampler` to stitch cross-partition frontier structure for shard-local non-temporal hetero graphs when sampling through a coordinator-backed feature source.

**Architecture:** Keep the public link sampler plan unchanged and add one stitched heterogeneous branch inside `PlanExecutor._sample_link_neighbors(...)`. That branch will lift typed link endpoints into global-id space, expand a typed visited frontier through coordinator incident-edge queries, build one stitched hetero graph with globally aligned typed node and edge tensors, and rebuild sampled `LinkPredictionRecord`s against that stitched graph.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add failing stitched hetero link sampling regressions

**Files:**
- Modify: `tests/data/test_link_neighbor_sampler.py`
- Modify: `tests/integration/test_foundation_partition_local.py`

**Step 1: Write the failing test**

Add regressions proving:

- a shard-local heterogeneous `LinkPredictionRecord` sampled through `LinkNeighborSampler` plus `LocalSamplingCoordinator` can cross a partition boundary and include remote typed nodes and relation edges
- the stitched hetero sampled graph keeps typed global `n_id`, typed global `e_id`, and optional fetched typed node/edge features aligned through the coordinator
- the public sampled hetero link-training loader path can consume the stitched hetero batch during link prediction

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched_hetero_link`
Expected: FAIL because heterogeneous link sampling still only expands through the shard-local graph.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched_hetero_link`
Expected: FAIL on missing remote typed stitched nodes or relation edges.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement stitched heterogeneous link expansion and materialization

**Files:**
- Modify: `vgl/dataloading/executor.py`
- Modify: `tests/data/test_link_neighbor_sampler.py`
- Modify: `tests/integration/test_foundation_partition_local.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched_hetero_link`
Expected: FAIL

**Step 3: Write minimal implementation**

Detect shard-local heterogeneous coordinator-backed link sampling inside `PlanExecutor._sample_link_neighbors(...)`, expand typed visited endpoints in global-id space through coordinator typed incident-edge queries with the same per-hop fanout semantics as stitched hetero node sampling, build one stitched heterogeneous graph with aligned typed node and edge tensors, rebuild sampled `LinkPredictionRecord`s against that graph, and store typed sampled indices in executor state for later feature overlays.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched_hetero_link`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Document stitched hetero link sampling scope and run branch verification

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review docs for places that still say heterogeneous link stitched sampling is future work.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py tests/integration/test_foundation_partition_local.py -k stitched_hetero_link`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document that shard-local non-temporal heterogeneous `LinkNeighborSampler` can now stitch cross-partition frontier structure through the local coordinator path, and state clearly that heterogeneous temporal stitched sampling remains future work.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add distributed stitched hetero link sampling"`
