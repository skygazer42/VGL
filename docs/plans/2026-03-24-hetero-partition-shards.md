# Heterogeneous Partition Shards Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the local distributed partition runtime so it can write, reload, and coordinate true multi-node-type heterogeneous graph shards.

**Architecture:** Keep the current local-first partition format and additive coordinator surface, but generalize partition ownership from one global `node` range to typed contiguous node ranges. Reuse the existing on-disk graph payload format, store per-type node-id tables inside each shard, and infer typed node routing from feature keys where possible.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add typed partition metadata coverage

**Files:**
- Modify: `tests/distributed/test_partition_metadata.py`
- Modify: `vgl/distributed/partition.py`

**Step 1: Write the failing test**

Add a test proving a manifest can resolve partition ownership per node type, expose per-type partition ranges, and preserve homogeneous behavior when `node_type` is omitted.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_partition_metadata.py -k typed`
Expected: FAIL because typed ownership helpers do not exist.

**Step 3: Write minimal implementation**

Extend `PartitionShard` and `PartitionManifest` additively with typed node-count/range support plus `owner(node_id, node_type="node")` and a typed range accessor.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/distributed/test_partition_metadata.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 2: Add heterogeneous partition-writer coverage and implementation

**Files:**
- Modify: `tests/distributed/test_partition_writer.py`
- Modify: `vgl/distributed/writer.py`

**Step 1: Write the failing test**

Add a regression proving `write_partitioned_graph(...)` can partition a graph with multiple node types, preserve per-type node ids and features, filter intra-partition typed edges correctly, and record typed node counts in the manifest.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_partition_writer.py -k hetero`
Expected: FAIL because the writer still rejects multi-node-type graphs.

**Step 3: Write minimal implementation**

Partition each node type independently, retain only edges whose typed endpoints belong to the current partition, relabel local source/destination spaces separately, and serialize per-type `node_ids` in the shard payload.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/distributed/test_partition_writer.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Add typed shard/coordinator routing support

**Files:**
- Modify: `tests/distributed/test_local_shard.py`
- Modify: `tests/distributed/test_sampling_coordinator.py`
- Modify: `vgl/distributed/shard.py`
- Modify: `vgl/distributed/coordinator.py`

**Step 1: Write the failing test**

Add tests proving a hetero shard rebuilds all typed node stores, maps local/global node ids per node type, reconstructs typed global edge indices, and lets the coordinator route/fetch node features for a requested node type.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_local_shard.py -k hetero tests/distributed/test_sampling_coordinator.py -k hetero`
Expected: FAIL because shard reload and coordinator routing still assume a single `node` namespace.

**Step 3: Write minimal implementation**

Teach `LocalGraphShard` to store per-type node-id tables and typed lookup maps. Add typed `global_to_local(...)`, `local_to_global(...)`, `partition_node_ids(...)`, and `route_node_ids(...)` behavior while keeping homogeneous defaults unchanged. Make `fetch_node_features(...)` infer node type from the feature key.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/distributed/test_local_shard.py tests/distributed/test_sampling_coordinator.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 4: Add hetero partition integration coverage

**Files:**
- Modify: `tests/integration/test_foundation_partition_local.py`

**Step 1: Write the failing test**

Add an integration regression that partitions a multi-node-type hetero graph, reloads typed shards, routes paper-node features through the coordinator, and runs one epoch of node classification using `NodeNeighborSampler` with `metadata["node_type"]`.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/integration/test_foundation_partition_local.py -k hetero`
Expected: FAIL because typed shard routing is not implemented yet.

**Step 3: Write minimal implementation**

Use the production changes from Tasks 1-3; only patch integration helpers if the new hetero path exposes a real gap.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/integration/test_foundation_partition_local.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 5: Refresh docs and run full regression

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`
- Modify: any touched modules as needed

**Step 1: Write the failing test**

No code test. Review docs for places that still describe the local partition path as single-node-type only.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/distributed/test_partition_metadata.py tests/distributed/test_partition_writer.py tests/distributed/test_local_shard.py tests/distributed/test_sampling_coordinator.py tests/integration/test_foundation_partition_local.py`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document that the local distributed runtime now supports homogeneous, temporal, single-node-type multi-relation, and true multi-node-type hetero partitions, with typed node routing via shard/coordinator helpers.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add heterogeneous partition shard support"`
