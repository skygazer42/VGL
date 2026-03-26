# Adjacency Query Ops Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add DGL-style adjacency query operators so VGL can retrieve inbound and outbound edges plus one-hop predecessors and successors through the public graph-ops layer.

**Architecture:** Extend `vgl.ops.query` with ordered adjacency selection helpers for `in_edges(...)`, `out_edges(...)`, `predecessors(...)`, and `successors(...)`, reuse the existing public `e_id` resolution model for edge-returning forms, and bridge the new APIs through `Graph` and `vgl.ops` exports. Keep the implementation read-only and layered on top of current edge-store order instead of introducing new graph views or adjacency caches.

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

### Task 1: Add failing adjacency-query regressions

**Files:**
- Modify: `tests/ops/test_query_ops.py`
- Modify: `tests/core/test_feature_backed_graph.py`
- Modify: `tests/core/test_graph_ops_api.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing test**

Add regressions proving:

- `in_edges(...)` and `out_edges(...)` support `uv`, `eid`, and `all` forms
- edge-returning forms resolve preserved public `e_id` on frontier subgraphs
- `predecessors(...)` and `successors(...)` preserve duplicates for parallel edges
- heterogeneous adjacency queries work on one selected relation
- invalid `form` values and out-of-range node ids fail clearly
- featureless storage-backed graphs can still query declared but isolated nodes
- `Graph` bridges and `vgl.ops` exports expose the new APIs

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_feature_backed_graph.py tests/core/test_graph_ops_api.py tests/test_package_layout.py -k "in_edges or out_edges or predecessors or successors"`
Expected: FAIL because the new adjacency-query functions, Graph bridges, and exports do not exist yet.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_feature_backed_graph.py tests/core/test_graph_ops_api.py tests/test_package_layout.py -k "in_edges or out_edges or predecessors or successors"`
Expected: FAIL on missing imports, missing Graph methods, or missing adjacency-query behavior.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement the adjacency query ops

**Files:**
- Modify: `vgl/ops/query.py`
- Modify: `vgl/ops/__init__.py`
- Modify: `tests/ops/test_query_ops.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_query_ops.py -k "in_edges or out_edges or predecessors or successors"`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement in `vgl/ops/query.py`:

- `in_edges(graph, v, *, form="uv", edge_type=None)`
- `out_edges(graph, u, *, form="uv", edge_type=None)`
- `predecessors(graph, v, *, edge_type=None)`
- `successors(graph, v, *, edge_type=None)`

Implementation rules:

- resolve the selected relation through the existing tuple-or-default convention
- accept scalar or iterable node collections for `in_edges(...)` and `out_edges(...)`
- support `form="uv"`, `form="eid"`, and `form="all"`
- surface public edge ids through `edata["e_id"]` when present
- preserve stable current edge order in every returned tensor
- preserve duplicate neighbors for parallel edges in `predecessors(...)` and `successors(...)`
- validate node ids against graph node counts and reject invalid `form` with `ValueError`

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_query_ops.py -k "in_edges or out_edges or predecessors or successors"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Bridge Graph methods and stable exports

**Files:**
- Modify: `vgl/graph/graph.py`
- Modify: `vgl/ops/__init__.py`
- Modify: `tests/core/test_graph_ops_api.py`
- Modify: `tests/test_package_layout.py`
- Modify: `tests/core/test_feature_backed_graph.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_feature_backed_graph.py tests/core/test_graph_ops_api.py tests/test_package_layout.py -k "in_edges or out_edges or predecessors or successors"`
Expected: FAIL

**Step 3: Write minimal implementation**

Add:

- `Graph.in_edges(...)`
- `Graph.out_edges(...)`
- `Graph.predecessors(...)`
- `Graph.successors(...)`

Update `vgl.ops` exports and package-layout expectations so the new functions are part of the stable public surface.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_feature_backed_graph.py tests/core/test_graph_ops_api.py tests/test_package_layout.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 4: Refresh docs and run the full regression suite

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review graph-ops docs for places that mention edge queries but still stop at `find_edges(...)`, `edge_ids(...)`, and `has_edges_between(...)`.

**Step 2: Run test to verify the code path state**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_feature_backed_graph.py tests/core/test_graph_ops_api.py tests/test_package_layout.py`
Expected: PASS if the adjacency query path is implemented cleanly.

**Step 3: Write minimal implementation**

Document:

- `in_edges(...)` and `out_edges(...)` forms
- `predecessors(...)` and `successors(...)` duplicate-preserving semantics
- continued use of `edata["e_id"]` as the stable public edge-id space
- featureless storage-backed node-count-sensitive adjacency queries

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add adjacency query ops"`
