# Storage-Backed Graph Node Count Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make featureless storage-backed graphs preserve declared node counts across base graph operations such as adjacency building, DGL export, and block construction.

**Architecture:** Extend `GraphStore` with `num_nodes(...)`, retain `graph_store` on `Graph.from_storage(...)`, and let `Graph` / `GraphView` resolve node counts from graph-store metadata when node features are absent. Keep all higher-level graph and block APIs unchanged.

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

### Task 1: Add failing storage-backed count regressions

**Files:**
- Modify: `tests/core/test_feature_backed_graph.py`
- Modify: `tests/ops/test_block_ops.py`
- Modify: `tests/compat/test_dgl_adapter.py`

**Step 1: Write the failing test**

Add regressions proving:

- `Graph.from_storage(...)` retains `graph_store` context
- featureless homogeneous storage-backed graphs preserve declared adjacency shape
- featureless heterogeneous storage-backed graphs preserve declared per-type adjacency shape
- `to_block(...)` works on featureless homogeneous storage-backed graphs using declared node counts
- featureless homogeneous storage-backed graphs export to DGL with declared `num_nodes()` intact

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/core/test_feature_backed_graph.py tests/ops/test_block_ops.py tests/compat/test_dgl_adapter.py -k "graph_store or featureless_storage_backed"`
Expected: FAIL because graph-store node counts are not yet retained or consulted by the graph abstraction.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/core/test_feature_backed_graph.py tests/ops/test_block_ops.py tests/compat/test_dgl_adapter.py -k "graph_store or featureless_storage_backed"`
Expected: FAIL on missing graph-store context or missing count fallback.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement graph-store-backed node count resolution

**Files:**
- Modify: `vgl/storage/graph_store.py`
- Modify: `vgl/distributed/store.py`
- Modify: `vgl/graph/graph.py`
- Modify: `vgl/graph/view.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/core/test_feature_backed_graph.py tests/ops/test_block_ops.py tests/compat/test_dgl_adapter.py -k "graph_store or featureless_storage_backed"`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- `GraphStore.num_nodes(node_type)` and `InMemoryGraphStore.num_nodes(...)`
- `LocalGraphStoreAdapter.num_nodes(...)` plus distributed protocol exposure
- `Graph.graph_store` retention in `from_storage(...)`
- graph-store fallback in `Graph._node_count(...)`
- graph-store fallback in `GraphView._node_count(...)`
- propagation of `graph_store` through `Graph.to(...)` / `pin_memory()`

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/core/test_feature_backed_graph.py tests/ops/test_block_ops.py tests/compat/test_dgl_adapter.py -k "graph_store or featureless_storage_backed"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Refresh docs and run full regression

**Files:**
- Modify: `docs/core-concepts.md`

**Step 1: Write the failing test**

No code test. Review the storage-backed graph docs for places that imply node counts come only from node tensors.

**Step 2: Run test to verify the code path state**

Run: `python -m pytest -q tests/core/test_feature_backed_graph.py tests/ops/test_block_ops.py tests/compat/test_dgl_adapter.py`
Expected: PASS if the storage-backed count changes are complete.

**Step 3: Write minimal implementation**

Document that featureless storage-backed graphs still preserve declared node counts through retained graph-store metadata, so adjacency views and count-only graph ops continue to work without fake node features.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs && git commit -m "feat: preserve storage-backed node counts"`
