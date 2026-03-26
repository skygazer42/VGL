# Graph Query And Reverse Ops Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add DGL-class edge-query primitives and a reversible graph transform so VGL can resolve endpoints, look up edge ids, test pair connectivity, and build reversed graphs through the public graph-ops layer.

**Architecture:** Introduce one compact query-focused ops module for `find_edges(...)`, `edge_ids(...)`, and `has_edges_between(...)`, extend the structural ops layer with `reverse(...)`, and bridge all four operations through `Graph`. Query resolution should treat `edata["e_id"]` as the stable public edge-id space when present so derived graphs keep consistent lookup semantics.

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

### Task 1: Add failing query and reverse regressions

**Files:**
- Create: `tests/ops/test_query_ops.py`
- Modify: `tests/core/test_feature_backed_graph.py`
- Modify: `tests/core/test_graph_ops_api.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing test**

Add regressions proving:

- `find_edges(...)` returns endpoints in requested edge-id order
- `find_edges(...)` resolves preserved public `e_id` on a derived graph
- `edge_ids(...)` returns stable edge ids for simple pairs and the first stable match for parallel edges
- `edge_ids(..., return_uv=True)` enumerates every match in pair order
- `has_edges_between(...)` returns scalar and vector connectivity answers
- `reverse(...)` flips homogeneous and heterogeneous relations
- `reverse(...)` honors `copy_ndata` / `copy_edata` while preserving `e_id`
- featureless storage-backed reversed graphs keep declared adjacency shape
- `Graph` bridges and `vgl.ops` exports expose the new API

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_feature_backed_graph.py tests/core/test_graph_ops_api.py tests/test_package_layout.py -k "find_edges or edge_ids or has_edges_between or reverse"`
Expected: FAIL because the new query/reverse functions, Graph bridges, and exports do not exist yet.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_feature_backed_graph.py tests/core/test_graph_ops_api.py tests/test_package_layout.py -k "find_edges or edge_ids or has_edges_between or reverse"`
Expected: FAIL on missing imports, missing Graph methods, or missing query/reverse behavior.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement the public query ops

**Files:**
- Create: `vgl/ops/query.py`
- Modify: `vgl/ops/__init__.py`
- Modify: `tests/ops/test_query_ops.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_query_ops.py -k "find_edges or edge_ids or has_edges_between"`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement in `vgl/ops/query.py`:

- `find_edges(graph, eids, *, edge_type=None)`
- `edge_ids(graph, u, v, *, return_uv=False, edge_type=None)`
- `has_edges_between(graph, u, v, *, edge_type=None)`

Implementation rules:

- resolve the selected relation through the existing tuple-or-default convention
- interpret `edata["e_id"]` as the public id space when present
- require paired `u` / `v` inputs to have equal length
- raise `ValueError` for unknown edge ids, absent queried pairs, or out-of-range node ids
- when `return_uv=False`, return one stable public edge id per queried pair and pick the first match for parallel edges
- when `return_uv=True`, enumerate every matching edge in pair order and stable edge order
- return Python `bool` for scalar `has_edges_between(...)` and boolean tensors otherwise

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_query_ops.py -k "find_edges or edge_ids or has_edges_between"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Implement reverse graph support and Graph bridges

**Files:**
- Modify: `vgl/ops/structure.py`
- Modify: `vgl/graph/graph.py`
- Modify: `vgl/ops/__init__.py`
- Modify: `tests/ops/test_query_ops.py`
- Modify: `tests/core/test_feature_backed_graph.py`
- Modify: `tests/core/test_graph_ops_api.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_feature_backed_graph.py tests/core/test_graph_ops_api.py tests/test_package_layout.py -k "reverse or find_edges or edge_ids or has_edges_between"`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- `reverse(graph, *, copy_ndata=True, copy_edata=False)` in `vgl.ops.structure`
- `Graph.find_edges(...)`
- `Graph.edge_ids(...)`
- `Graph.has_edges_between(...)`
- `Graph.reverse(...)`

Implementation rules:

- reverse homogeneous edge indices by swapping source and destination rows
- reverse heterogeneous relation keys to `(dst_type, relation_name, src_type)`
- preserve node stores when `copy_ndata=True`, otherwise return empty node stores for every node type
- preserve `e_id` even when `copy_edata=False`
- when `copy_edata=True`, keep edge-aligned tensors in the same stable edge order
- preserve graph-level storage context needed for node counts and shared feature access
- export the new ops from `vgl.ops`

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

No code test. Review the graph-ops docs for places that describe subgraphs and structure rewrites but not the new query/reverse layer.

**Step 2: Run test to verify the code path state**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_feature_backed_graph.py tests/core/test_graph_ops_api.py tests/test_package_layout.py`
Expected: PASS if the new graph-query layer is implemented cleanly.

**Step 3: Write minimal implementation**

Document:

- `find_edges(...)`, `edge_ids(...)`, and `has_edges_between(...)` as the core edge-query surface
- `reverse(...)` as the public reversed-graph transform
- `edata["e_id"]` as the stable public edge-id contract for derived graphs
- the new `Graph` convenience methods and featureless storage-backed reverse behavior

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add graph query and reverse ops"`
