# Manifest Feature Shape Metadata Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make partition feature shape queries resolve from manifest metadata so lazy distributed stores and store-backed coordinators stay payload-free for empty fetches and shape inspection.

**Architecture:** Extend `PartitionShard.metadata` with serialized node/edge feature-shape maps, emit those maps from the partition writer, and teach the lazy partition feature adapter to answer `shape(...)` from metadata before touching `part-*.pt`.

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

### Task 1: Add failing regressions for manifest-only shape paths

**Files:**
- Modify: `tests/distributed/test_partition_metadata.py`
- Modify: `tests/distributed/test_store_protocol.py`
- Modify: `tests/distributed/test_sampling_coordinator.py`

**Step 1: Write the failing test**

Add regressions proving:

- partition manifests round-trip node and edge feature-shape metadata
- `load_partitioned_stores(...).shape(...)` stays at zero `torch.load(...)` calls
- empty `StoreBackedSamplingCoordinator.fetch_node_features(...)` requests stay manifest-only

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_partition_metadata.py tests/distributed/test_store_protocol.py tests/distributed/test_sampling_coordinator.py -k "shape or lazy"`
Expected: FAIL because partition metadata does not yet persist feature shapes and lazy feature stores still load payloads for `shape(...)`.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/distributed/test_partition_metadata.py tests/distributed/test_store_protocol.py tests/distributed/test_sampling_coordinator.py -k "shape or lazy"`
Expected: FAIL on missing metadata round-trip or unexpected payload loads.

**Step 5: Commit**

Do not commit yet.

### Task 2: Add partition metadata support for feature shapes

**Files:**
- Modify: `vgl/distributed/partition.py`
- Modify: `vgl/distributed/writer.py`
- Modify: `tests/distributed/test_partition_metadata.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_partition_metadata.py -k shape`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- normalization and serialization for `node_feature_shapes`
- normalization and serialization for `edge_feature_shapes`
- partition-writer emission of local node and edge feature shapes

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/distributed/test_partition_metadata.py -k shape`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Make lazy feature-store shape queries metadata-only

**Files:**
- Modify: `vgl/distributed/store.py`
- Modify: `tests/distributed/test_store_protocol.py`
- Modify: `tests/distributed/test_sampling_coordinator.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_store_protocol.py tests/distributed/test_sampling_coordinator.py -k "shape or lazy"`
Expected: FAIL

**Step 3: Write minimal implementation**

Teach the lazy partition feature adapter to resolve partition-local shapes from manifest metadata and keep empty coordinator feature fetches payload-free until real tensor values are requested.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/distributed/test_store_protocol.py tests/distributed/test_sampling_coordinator.py -k "shape or lazy"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 4: Run focused and full verification

**Files:**
- Modify: any touched files from prior tasks as needed

**Step 1: Re-read touched code paths**

Confirm the new metadata keys remain backward-compatible for manifests that do not declare feature shapes.

**Step 2: Run focused verification**

Run: `python -m pytest -q tests/distributed/test_partition_metadata.py tests/distributed/test_store_protocol.py tests/distributed/test_sampling_coordinator.py tests/data/test_node_neighbor_sampler.py tests/data/test_temporal_neighbor_sampler.py tests/integration/test_foundation_partition_local.py`
Expected: PASS

**Step 3: Run full verification**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 4: Prepare integration**

Merge the worktree branch back to `main`, push `origin/main`, and remove the temporary worktree and branch so only `main` remains.

**Step 5: Commit**

Run: `git add vgl tests docs && git commit -m "feat: add manifest-backed partition feature shapes"`
