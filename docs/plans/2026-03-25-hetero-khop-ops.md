# Heterogeneous K-Hop Ops Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `khop_nodes(...)` and `khop_subgraph(...)` so relation-local heterogeneous graph ops work through the current public API.

**Architecture:** Preserve the current homogeneous tensor-based behavior, but teach bipartite heterogeneous relations selected through `edge_type` to consume seeds keyed by node type and return node ids keyed by node type. Keep `khop_subgraph(...)` layered on top of `node_subgraph(...)`.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add failing hetero k-hop regression coverage

**Files:**
- Modify: `tests/ops/test_khop_ops.py`

**Step 1: Write the failing test**

Add regressions proving:

- `khop_nodes(...)` supports relation-local outbound bipartite hetero expansion
- `khop_nodes(...)` supports relation-local inbound bipartite hetero expansion
- `khop_subgraph(...)` returns a relabelled hetero subgraph for that node set
- bipartite hetero `khop_nodes(...)` rejects flat tensor seeds and requires node-type-keyed seeds

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_khop_ops.py -k hetero`
Expected: FAIL because `khop_nodes(...)` still assumes one flat node-id tensor.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/ops/test_khop_ops.py -k hetero`
Expected: FAIL on missing hetero k-hop support.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement relation-local hetero k-hop support

**Files:**
- Modify: `vgl/ops/khop.py`
- Modify: `tests/ops/test_khop_ops.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_khop_ops.py -k hetero`
Expected: FAIL

**Step 3: Write minimal implementation**

Keep same-type relations on the existing tensor path, but for bipartite heterogeneous relations require seeds keyed by node type, expand along the selected relation according to `direction`, and return per-type node ids that `khop_subgraph(...)` can pass through to `node_subgraph(...)`.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_khop_ops.py -k hetero`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Refresh docs and run branch verification

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`
- Create: `docs/plans/2026-03-25-hetero-khop-ops.md`
- Create: `docs/plans/2026-03-25-hetero-khop-ops-design.md`

**Step 1: Write the failing test**

No code test. Review docs for places that still describe `k-hop` expansion as a flat homogeneous-only node-id flow.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/core/test_graph_ops_api.py tests/ops/test_khop_ops.py tests/ops/test_subgraph_ops.py tests/ops/test_compact_ops.py tests/data/test_neighbor_expansion_stage.py`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document the relation-local heterogeneous `k-hop` contract, including per-type seeds for bipartite relations.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: support hetero k-hop graph ops"`
