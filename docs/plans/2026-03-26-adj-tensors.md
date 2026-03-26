# Adjacency Tensor Export Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add DGL-style `adj_tensors(...)` exports for COO, CSR, and CSC adjacency structure.

**Architecture:** Extend `vgl.ops.query` with one adjacency-tensor exporter that resolves relations like the other query ops, uses public `e_id` order as the visible edge identity model, and derives compressed tensors with stable structural sorting for CSR and CSC. Bridge the new op through `Graph` and `vgl.ops`, then document the new raw-export surface.

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

### Task 1: Add failing adjacency-tensor regressions

**Files:**
- Modify: `tests/ops/test_query_ops.py`
- Modify: `tests/core/test_graph_ops_api.py`
- Modify: `tests/core/test_feature_backed_graph.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing test**

Add regressions proving:

- `adj_tensors(...)` supports `layout="coo"`, `"csr"`, and `"csc"`
- COO output follows public `e_id` ordering on derived graphs
- CSR and CSC outputs return structural tensors plus aligned public edge ids
- duplicate edges remain stable under compression
- heterogeneous graphs work with explicit `edge_type`
- featureless storage-backed graphs expose raw adjacency tensors
- `Graph` bridges and `vgl.ops` exports expose the new API

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "adj_tensors"`
Expected: FAIL because the new op, Graph bridge, and exports do not exist yet.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "adj_tensors"`
Expected: FAIL on missing imports, missing Graph methods, or missing query behavior.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement adjacency-tensor query behavior

**Files:**
- Modify: `vgl/ops/query.py`
- Modify: `tests/ops/test_query_ops.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_query_ops.py -k "adj_tensors"`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement in `vgl/ops/query.py`:

- `adj_tensors(graph, layout="coo", *, edge_type=None)`

Implementation rules:

- normalize layout from string or `SparseLayout`
- resolve the target relation with the same helper as other query ops
- return COO tensors in public-`e_id` order when present
- return CSR and CSC pointer/index tensors in stable structural order
- expose aligned public edge ids as the third tensor for CSR and CSC
- reject unsupported layouts clearly

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_query_ops.py -k "adj_tensors"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Bridge Graph methods and stable exports

**Files:**
- Modify: `vgl/graph/graph.py`
- Modify: `vgl/ops/__init__.py`
- Modify: `tests/core/test_graph_ops_api.py`
- Modify: `tests/core/test_feature_backed_graph.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "adj_tensors"`
Expected: FAIL

**Step 3: Write minimal implementation**

Add:

- `Graph.adj_tensors(...)`

Update `vgl.ops` exports and namespace expectations so `adj_tensors` becomes part of the stable public surface.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "adj_tensors"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 4: Refresh docs and run the full regression suite

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review graph-query and sparse-layer docs for places that mention adjacency views but not raw adjacency tensor export.

**Step 2: Run test to verify the code path state**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "adj_tensors"`
Expected: PASS if the adjacency-tensor path is implemented cleanly.

**Step 3: Write minimal implementation**

Document:

- supported `coo` / `csr` / `csc` return signatures
- public-`e_id` ordering behavior
- the relation-selection pattern for heterogeneous graphs

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add adjacency tensor exports"`
