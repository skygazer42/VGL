# Cardinality And Full-Edge Query Ops Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add DGL-style graph cardinality helpers plus relation-local full-edge enumeration so VGL can report node/edge totals and enumerate every edge through the public graph API.

**Architecture:** Extend `vgl.ops.query` with lightweight count helpers and one ordered full-edge selector. Reuse the existing public `e_id` model for returned edge ids and `order="eid"` sorting, keep heterogeneous counts total-aware when no type is specified, and bridge everything through `Graph` and `vgl.ops` exports.

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

### Task 1: Add failing cardinality and full-edge regressions

**Files:**
- Modify: `tests/ops/test_query_ops.py`
- Modify: `tests/core/test_graph_ops_api.py`
- Modify: `tests/core/test_feature_backed_graph.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing test**

Add regressions proving:

- `num_nodes(...)` / `number_of_nodes(...)` return total or per-type counts
- `num_edges(...)` / `number_of_edges(...)` return total or per-relation counts
- `all_edges(...)` supports `uv`, `eid`, and `all` forms
- `all_edges(order="eid")` sorts by public `e_id`
- `all_edges(order="srcdst")` sorts lexicographically by endpoints
- multi-relation graphs require `edge_type` for `all_edges(...)`
- featureless storage-backed graphs preserve declared node totals
- `Graph` bridges and `vgl.ops` exports expose the new APIs

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "num_nodes or num_edges or all_edges or number_of_nodes or number_of_edges"`
Expected: FAIL because the new count helpers, aliases, and full-edge query APIs do not exist yet.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "num_nodes or num_edges or all_edges or number_of_nodes or number_of_edges"`
Expected: FAIL on missing imports, missing Graph methods, or missing cardinality/full-edge behavior.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement the count and full-edge query ops

**Files:**
- Modify: `vgl/ops/query.py`
- Modify: `vgl/ops/__init__.py`
- Modify: `tests/ops/test_query_ops.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_query_ops.py -k "num_nodes or num_edges or all_edges or number_of_nodes or number_of_edges"`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement in `vgl/ops/query.py`:

- `num_nodes(graph, node_type=None)`
- `number_of_nodes(graph, node_type=None)`
- `num_edges(graph, edge_type=None)`
- `number_of_edges(graph, edge_type=None)`
- `all_edges(graph, *, form="uv", order="eid", edge_type=None)`

Implementation rules:

- omitted `node_type` returns the total declared node count across all node types
- omitted `edge_type` in count helpers returns the total edge count across all relations
- omitted `edge_type` in `all_edges(...)` only works when the graph has one relation; otherwise fail clearly
- `all_edges(...)` reuses public `e_id` semantics from the existing query layer
- `order="eid"` sorts by public edge ids
- `order="srcdst"` sorts by source, destination, then public edge id
- invalid `form` or `order` raises `ValueError`

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_query_ops.py -k "num_nodes or num_edges or all_edges or number_of_nodes or number_of_edges"`
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

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "num_nodes or num_edges or all_edges or number_of_nodes or number_of_edges"`
Expected: FAIL

**Step 3: Write minimal implementation**

Add:

- `Graph.num_nodes(...)`
- `Graph.number_of_nodes(...)`
- `Graph.num_edges(...)`
- `Graph.number_of_edges(...)`
- `Graph.all_edges(...)`

Update `vgl.ops` exports and namespace expectations so the new helpers become part of the stable public surface.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "num_nodes or num_edges or all_edges or number_of_nodes or number_of_edges"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 4: Refresh docs and run the full regression suite

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review public docs for graph-query descriptions that still force users into `graph.schema`, `graph.nodes`, or `graph.edge_index` for basic counts and full-edge enumeration.

**Step 2: Run test to verify the code path state**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "num_nodes or num_edges or all_edges or number_of_nodes or number_of_edges"`
Expected: PASS if the new count and full-edge query path is implemented cleanly.

**Step 3: Write minimal implementation**

Document:

- total-versus-type-specific node and edge count behavior
- `all_edges(...)` forms and ordering
- continued use of public `e_id` for returned ids and `order="eid"`

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add graph cardinality and all-edge queries"`
