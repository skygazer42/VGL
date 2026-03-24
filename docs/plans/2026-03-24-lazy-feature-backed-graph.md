# Lazy Feature-Backed Graph Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `Graph.from_storage(...)` resolve node and edge feature tensors lazily so storage-backed graphs stop fetching every feature at construction time.

**Architecture:** Keep the existing `Graph` / `NodeStore` / `EdgeStore` API stable, but change store construction from eager tensor materialization to lazy feature descriptors backed by `FeatureStore`. `edge_index` remains eager structural data, while non-structural features resolve on first access and cache thereafter.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add regression coverage for lazy storage-backed graph construction

**Files:**
- Modify: `tests/core/test_feature_backed_graph.py`

**Step 1: Write the failing test**

Add a test proving `Graph.from_storage(...)` does not call `FeatureStore.fetch(...)` during construction and only fetches a node feature when that feature is first accessed.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/core/test_feature_backed_graph.py -k lazy`
Expected: FAIL because the current implementation eagerly fetches all features during graph construction.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/core/test_feature_backed_graph.py -k lazy`
Expected: FAIL with eager fetch counts.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement lazy node and edge feature resolution

**Files:**
- Modify: `vgl/graph/stores.py`
- Modify: `tests/core/test_feature_backed_graph.py`

**Step 1: Write the failing test**

Use the test from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/core/test_feature_backed_graph.py -k lazy`
Expected: FAIL

**Step 3: Write minimal implementation**

Teach `NodeStore.from_feature_store(...)` and `EdgeStore.from_storage(...)` to register lazy feature names and fetch them on first access, caching the fetched tensors for subsequent reads.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/core/test_feature_backed_graph.py -k lazy`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Add mmap-backed integration regression coverage

**Files:**
- Modify: `tests/integration/test_foundation_large_graph_flow.py`

**Step 1: Write the failing test**

Add an integration test that builds a storage-backed graph with `MmapTensorStore`, routes it through the existing sampled loader and trainer path, and proves the public large-graph flow still works.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/integration/test_foundation_large_graph_flow.py -k mmap`
Expected: FAIL if the current storage-backed flow still assumes eager in-memory tensors or breaks on mmap-backed features.

**Step 3: Write minimal implementation**

Use the lazy store behavior from Task 2; only patch production code further if the integration reveals a real gap.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/integration/test_foundation_large_graph_flow.py -k mmap`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 4: Refresh docs and run full regression

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`
- Modify: any touched modules as needed

**Step 1: Write the failing test**

No code test. Review storage-backed graph docs for places that still imply eager feature materialization.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/core/test_feature_backed_graph.py tests/integration/test_foundation_large_graph_flow.py`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document that storage-backed graphs now resolve features lazily and work naturally with `MmapTensorStore`.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add lazy feature-backed graph loading"`
