# Distributed Runtime Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand the local-first distributed runtime so shards and coordinators can serve partition graph structure, not just routed feature fetches.

**Architecture:** Keep the current local partition format intact and build on top of it. `LocalGraphShard` will expose globalized structure helpers, and `LocalSamplingCoordinator` will add partition-scoped graph queries that delegate to the loaded shards while preserving the existing feature-routing behavior.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add shard regression coverage for local/global node-id conversion

**Files:**
- Modify: `tests/distributed/test_local_shard.py`

**Step 1: Write the failing test**

Add tests proving a shard can map local node ids back to global ids and can recover a partition-global edge index from its local stored graph.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_local_shard.py -k global`
Expected: FAIL because the shard does not expose local-to-global or global edge helpers.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/distributed/test_local_shard.py -k global`
Expected: FAIL with missing attribute errors.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement shard global-structure helpers

**Files:**
- Modify: `vgl/distributed/shard.py`
- Modify: `tests/distributed/test_local_shard.py`

**Step 1: Write the failing test**

Use the test from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_local_shard.py -k global`
Expected: FAIL

**Step 3: Write minimal implementation**

Add `local_to_global()` and `global_edge_index()` on `LocalGraphShard`, derived from the shard's `node_ids` and local edge index.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/distributed/test_local_shard.py -k global`
Expected: PASS

**Step 5: Commit**

Run: `git add tests/distributed/test_local_shard.py vgl/distributed/shard.py && git commit -m "feat: expose shard global structure helpers"`

### Task 3: Add coordinator regression coverage for partition graph queries

**Files:**
- Modify: `tests/distributed/test_sampling_coordinator.py`

**Step 1: Write the failing test**

Add tests showing that `LocalSamplingCoordinator` can return partition node ids, partition edge indices in both local/global forms, and partition adjacency in different layouts.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_sampling_coordinator.py -k partition`
Expected: FAIL because the coordinator does not expose partition graph-query helpers.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/distributed/test_sampling_coordinator.py -k partition`
Expected: FAIL with missing attribute errors.

**Step 5: Commit**

Do not commit yet.

### Task 4: Implement coordinator partition graph queries

**Files:**
- Modify: `vgl/distributed/coordinator.py`
- Modify: `tests/distributed/test_sampling_coordinator.py`

**Step 1: Write the failing test**

Use the test from Task 3.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_sampling_coordinator.py -k partition`
Expected: FAIL

**Step 3: Write minimal implementation**

Add `partition_node_ids(partition_id)`, `fetch_partition_edge_index(partition_id, global_ids=False)`, and `fetch_partition_adjacency(partition_id, layout=...)` to `LocalSamplingCoordinator`. Keep them partition-scoped and local-first.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/distributed/test_sampling_coordinator.py -k partition`
Expected: PASS

**Step 5: Commit**

Run: `git add tests/distributed/test_sampling_coordinator.py vgl/distributed/coordinator.py && git commit -m "feat: add coordinator graph structure queries"`

### Task 5: Refresh docs and run full regression

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`
- Modify: any touched modules as needed

**Step 1: Write the failing test**

No code test. Review distributed docs for places that still describe the coordinator as feature-only.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_local_shard.py tests/distributed/test_sampling_coordinator.py`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document that the local distributed runtime now exposes partition graph structure queries in addition to node-feature routing.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: expand distributed runtime queries"`
