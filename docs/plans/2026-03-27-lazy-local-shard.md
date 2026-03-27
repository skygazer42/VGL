# Lazy LocalGraphShard Loading Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `LocalGraphShard.from_partition_dir(...)` construct shards from manifest metadata only, deferring partition payload deserialization until the first real feature, graph, or boundary-edge access.

**Architecture:** Reuse the existing lazy partition bundle cache and adapters from `vgl.distributed.store`, derive shard node ids and edge-id mappings from manifest metadata, and rebuild `GraphSchema` from partition metadata plus manifest-level `time_attr`. Extend the shared cache to retain boundary-edge payloads so shard and coordinator paths still share one payload load.

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

### Task 1: Add failing lazy-shard regressions

**Files:**
- Modify: `tests/distributed/test_local_shard.py`

**Step 1: Write the failing test**

Add regressions proving:

- `LocalGraphShard.from_partition_dir(...)` does not call `torch.load(...)` during construction
- manifest-only shard methods still work before payload load
- first graph/feature/boundary access loads exactly one payload and reuses it
- temporal shards preserve `schema.time_attr` through the lazy path

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_local_shard.py -k "lazy or temporal"`
Expected: FAIL because shard construction still eagerly loads the partition payload.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/distributed/test_local_shard.py -k "lazy or temporal"`
Expected: FAIL on eager load counts or missing manifest-only behavior.

**Step 5: Commit**

Do not commit yet.

### Task 2: Make shard construction manifest-only

**Files:**
- Modify: `vgl/distributed/shard.py`
- Modify: `vgl/distributed/store.py`
- Modify: `vgl/distributed/partition.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_local_shard.py -k "lazy"`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- manifest-derived `node_ids_by_type`
- manifest-derived edge-id mappings that avoid eager graph access
- lazy shard `feature_store`, `graph_store`, `graph`, and `boundary_edge_data_by_type` backed by the shared partition bundle cache
- manifest-driven shard schema reconstruction

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/distributed/test_local_shard.py -k "lazy"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Persist temporal metadata for lazy shards

**Files:**
- Modify: `vgl/distributed/writer.py`
- Modify: `tests/distributed/test_local_shard.py`

**Step 1: Write the failing test**

Use the temporal lazy regression from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_local_shard.py -k "temporal"`
Expected: FAIL because the lazy shard path cannot reconstruct `time_attr` from manifest metadata.

**Step 3: Write minimal implementation**

Persist `time_attr` in manifest metadata and consume it when rebuilding shard schema.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/distributed/test_local_shard.py -k "temporal"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 4: Run focused and full verification

**Files:**
- Modify: any touched files from prior tasks as needed

**Step 1: Focused verification**

Run: `python -m pytest -q tests/distributed/test_local_shard.py tests/distributed/test_sampling_coordinator.py tests/data/test_feature_fetch_stage.py tests/integration/test_foundation_partition_local.py`
Expected: PASS

**Step 2: Full verification**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 3: Prepare integration**

Merge the worktree branch back to `main`, push `origin/main`, and remove the temporary worktree and branch so only `main` remains locally and remotely.

**Step 4: Commit**

Run: `git add vgl tests docs && git commit -m "feat: lazy-load local graph shards"`
