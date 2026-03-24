# On-Disk Dataset Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand `OnDiskGraphDataset` so it can round-trip heterogeneous and temporal graphs in addition to existing homogeneous graphs.

**Architecture:** Keep the current `manifest.json` + `graphs.pt` container layout, but generalize graph payload serialization to store typed node dictionaries, typed edge dictionaries, and optional temporal metadata. Reconstruct graphs through `Graph.hetero(...)` or `Graph.temporal(...)` so the dataset API stays stable while the underlying format supports more graph kinds.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add heterogeneous on-disk round-trip regression coverage

**Files:**
- Modify: `tests/data/test_ondisk_dataset.py`

**Step 1: Write the failing test**

Add a test proving `OnDiskGraphDataset.write(...)` and `OnDiskGraphDataset(...)` round-trip a heterogeneous graph with multiple node types, a typed edge relation, and edge features.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_ondisk_dataset.py -k heterogeneous`
Expected: FAIL because the serializer rejects non-homogeneous graphs.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/data/test_ondisk_dataset.py -k heterogeneous`
Expected: FAIL with the current homogeneous-only serialization error.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement heterogeneous on-disk graph serialization

**Files:**
- Modify: `vgl/data/ondisk.py`
- Modify: `tests/data/test_ondisk_dataset.py`

**Step 1: Write the failing test**

Use the test from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_ondisk_dataset.py -k heterogeneous`
Expected: FAIL

**Step 3: Write minimal implementation**

Generalize `serialize_graph(...)` and `deserialize_graph(...)` so they preserve typed node stores and typed edge stores for non-temporal graphs.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_ondisk_dataset.py -k heterogeneous`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Add temporal on-disk round-trip regression coverage

**Files:**
- Modify: `tests/data/test_ondisk_dataset.py`

**Step 1: Write the failing test**

Add a test proving a temporal graph round-trips with `time_attr`, `edge_index`, and timestamp edge data intact.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_ondisk_dataset.py -k temporal`
Expected: FAIL because the serializer rejects temporal graphs.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/data/test_ondisk_dataset.py -k temporal`
Expected: FAIL with the current non-temporal-only serialization error.

**Step 5: Commit**

Do not commit yet.

### Task 4: Implement temporal on-disk graph serialization

**Files:**
- Modify: `vgl/data/ondisk.py`
- Modify: `tests/data/test_ondisk_dataset.py`

**Step 1: Write the failing test**

Use the test from Task 3.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_ondisk_dataset.py -k temporal`
Expected: FAIL

**Step 3: Write minimal implementation**

Extend the generalized on-disk payload to preserve `time_attr` and reconstruct temporal graphs through `Graph.temporal(...)`.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_ondisk_dataset.py -k temporal`
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

No code test. Review docs for places that still imply on-disk datasets are only useful for homogeneous graphs.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_ondisk_dataset.py`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document that `OnDiskGraphDataset` can now persist homogeneous, heterogeneous, and temporal graphs through one manifest-backed dataset format.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: expand on-disk dataset graph support"`
