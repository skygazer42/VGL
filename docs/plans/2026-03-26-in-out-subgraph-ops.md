# Inbound And Outbound Frontier Subgraph Ops Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add DGL-class `in_subgraph(...)` and `out_subgraph(...)` graph ops that preserve node space while filtering edges by inbound or outbound frontier.

**Architecture:** Extend `vgl.ops.subgraph` with two non-mutating frontier extraction helpers, preserve the original node stores and graph-level storage context on outputs, and export the new ops through both `vgl.ops` and `Graph` convenience methods. Heterogeneous graphs should compose one filtered subgraph per relation back into one graph with the same schema as the input.

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

### Task 1: Add failing frontier-subgraph regressions

**Files:**
- Modify: `tests/ops/test_subgraph_ops.py`
- Modify: `tests/core/test_feature_backed_graph.py`

**Step 1: Write the failing test**

Add regressions proving:

- `in_subgraph(...)` on a homogeneous graph keeps the original node space and filters edges by destination frontier
- `out_subgraph(...)` on a homogeneous graph keeps the original node space and filters edges by source frontier
- both ops expose raw/public selected edge ids as `edata["e_id"]`
- heterogeneous frontier extraction composes across all relations using node-type keyed frontiers
- ambiguous heterogeneous tensor input fails with a clear `ValueError`
- featureless storage-backed graphs keep declared adjacency shape after `in_subgraph(...)` and `out_subgraph(...)`

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_subgraph_ops.py tests/core/test_feature_backed_graph.py -k "in_subgraph or out_subgraph"`
Expected: FAIL because the new frontier-subgraph functions and graph bridges do not exist yet.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/ops/test_subgraph_ops.py tests/core/test_feature_backed_graph.py -k "in_subgraph or out_subgraph"`
Expected: FAIL on missing imports, missing methods, or missing frontier-subgraph behavior.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement in_subgraph and out_subgraph in `vgl.ops`

**Files:**
- Modify: `vgl/ops/subgraph.py`
- Modify: `vgl/ops/__init__.py`
- Modify: `tests/ops/test_subgraph_ops.py`
- Modify: `tests/core/test_feature_backed_graph.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_subgraph_ops.py tests/core/test_feature_backed_graph.py -k "in_subgraph or out_subgraph"`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement in `vgl/ops/subgraph.py`:

- `in_subgraph(graph, nodes)`
- `out_subgraph(graph, nodes)`

Implementation rules:

- normalize homogeneous input as one node frontier for `"node"`
- require node-type keyed frontiers when the graph has more than one node type
- for `in_subgraph(...)`, keep edges whose destination id is in the selected frontier for that relation's destination node type
- for `out_subgraph(...)`, keep edges whose source id is in the selected frontier for that relation's source node type
- preserve the input graph's node stores, schema, `feature_store`, and `graph_store`
- rebuild edge stores with sliced edge-aligned tensors and a fresh adjacency cache
- always set `edata["e_id"]` to the selected public/raw edge ids in frontier order
- keep all original edge types on heterogeneous outputs, even when one relation contributes zero edges

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_subgraph_ops.py tests/core/test_feature_backed_graph.py -k "in_subgraph or out_subgraph"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Bridge Graph methods and stable exports

**Files:**
- Modify: `vgl/graph/graph.py`
- Modify: `tests/core/test_graph_ops_api.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing test**

Extend coverage to prove:

- `graph.in_subgraph(...)` delegates correctly
- `graph.out_subgraph(...)` delegates correctly
- `vgl.ops.__all__` and package-layout expectations expose the new functions

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/core/test_graph_ops_api.py tests/test_package_layout.py -k "in_subgraph or out_subgraph or ops_all"`
Expected: FAIL because the `Graph` bridges and exports are incomplete.

**Step 3: Write minimal implementation**

Add `Graph.in_subgraph(...)` and `Graph.out_subgraph(...)` convenience methods that delegate into `vgl.ops`, and update `vgl/ops/__init__.py` plus package-layout expectations so the new functions are part of the stable exported surface.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_subgraph_ops.py tests/core/test_feature_backed_graph.py tests/core/test_graph_ops_api.py tests/test_package_layout.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 4: Refresh docs and run full regression

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review graph-ops docs for places that still list only induced subgraphs and k-hop extraction.

**Step 2: Run test to verify the code path state**

Run: `python -m pytest -q tests/ops/test_subgraph_ops.py tests/core/test_feature_backed_graph.py tests/core/test_graph_ops_api.py tests/test_package_layout.py`
Expected: PASS if the frontier-subgraph path is implemented cleanly.

**Step 3: Write minimal implementation**

Document:

- `in_subgraph(...)` as inbound frontier extraction
- `out_subgraph(...)` as outbound frontier extraction
- preserved node-space semantics versus `node_subgraph(...)`
- the new `Graph` convenience methods and the fact that storage-backed graphs retain node-count context through these ops

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add frontier subgraph ops"`
