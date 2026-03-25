# Distributed Sampled Feature Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make sampled node/link/temporal feature fetch work correctly when the source graph is a partition shard and the feature source is a `LocalSamplingCoordinator`.

**Architecture:** Keep the current sampler and loader APIs intact. Extend `PlanExecutor` so neighbor-expansion stages can retain both local materialization ids and global fetch ids, then make routed feature sources consume the global view while direct stores keep the existing local-index semantics.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add failing node-sampling regression coverage

**Files:**
- Modify: `tests/data/test_node_neighbor_sampler.py`
- Modify: `tests/integration/test_foundation_partition_local.py`

**Step 1: Write the failing test**

Add regressions proving:

- `Loader(..., sampler=NodeNeighborSampler(...), feature_store=coordinator)` overlays correct global node and edge features onto sampled shard subgraphs
- the public sampled training path can consume coordinator-backed feature fetch without a custom routed loader wrapper

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py -k coordinator tests/integration/test_foundation_partition_local.py -k coordinator`
Expected: FAIL because sampled shard plans still hand local staged ids to the coordinator-backed fetch path.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py -k coordinator tests/integration/test_foundation_partition_local.py -k coordinator`
Expected: FAIL on misaligned fetched features or the remaining wrapper-only integration path.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement routed sampled-id alignment in the executor

**Files:**
- Modify: `vgl/dataloading/executor.py`
- Modify: `tests/data/test_node_neighbor_sampler.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py -k coordinator tests/integration/test_foundation_partition_local.py -k coordinator`
Expected: FAIL

**Step 3: Write minimal implementation**

Teach neighbor-expansion stages to retain global node and edge ids alongside the current local materialization ids whenever the source graph exposes `n_id` / `e_id`. Teach feature-fetch stages to prefer those global ids only for routed sources that use `fetch_node_features(...)` / `fetch_edge_features(...)`, while direct `.fetch(...)` stores keep current behavior.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py -k coordinator tests/integration/test_foundation_partition_local.py -k coordinator`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Add and verify link/temporal sampled regressions

**Files:**
- Modify: `tests/data/test_link_neighbor_sampler.py`
- Modify: `tests/data/test_temporal_neighbor_sampler.py`
- Modify: `vgl/dataloading/executor.py`

**Step 1: Write the failing test**

Add regressions proving sampled link and temporal shard subgraphs can fetch routed node and edge features through the coordinator without double-translating their already-global `n_id` / `e_id` state.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py -k coordinator tests/data/test_temporal_neighbor_sampler.py -k coordinator`
Expected: FAIL if the executor alignment path still mishandles sampled-record fetch indices.

**Step 3: Write minimal implementation**

Use the production changes from Task 2; only patch executor helpers if sampled-record stages expose a remaining distinction between local and global staged ids.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py -k coordinator tests/data/test_temporal_neighbor_sampler.py -k coordinator`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 4: Refresh docs and run branch verification

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`
- Modify: any touched modules as needed

**Step 1: Write the failing test**

No code test. Review docs for places that still imply coordinator-backed sampled loading needs manual local/global feature glue.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_node_neighbor_sampler.py tests/data/test_link_neighbor_sampler.py tests/data/test_temporal_neighbor_sampler.py tests/integration/test_foundation_partition_local.py`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document that shard graphs can stay local for structure while coordinator-backed feature fetch stages align through `n_id` / `e_id` automatically in the public loader path.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: align distributed sampled feature fetch"`
