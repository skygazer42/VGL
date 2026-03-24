# Mmap Tensor Store Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn `MmapTensorStore` into a true memory-mapped tensor store while preserving its current public API and legacy file compatibility.

**Architecture:** Newly written stores will use a raw tensor buffer at the provided path plus a JSON metadata sidecar that records shape and dtype. Reads will prefer metadata-backed `numpy.memmap(...)` access and fall back to legacy `torch.load(...)` files when the sidecar is missing.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pytest

---

### Task 1: Add mmap sidecar regression coverage

**Files:**
- Modify: `tests/storage/test_mmap_store.py`

**Step 1: Write the failing test**

Add a test proving `MmapTensorStore.save(...)` writes a metadata sidecar and still supports fetch, shape, and dtype queries through the existing API.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/storage/test_mmap_store.py -k sidecar`
Expected: FAIL because the current implementation only writes a `torch.save(...)` file and no metadata sidecar.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/storage/test_mmap_store.py -k sidecar`
Expected: FAIL with missing sidecar expectations.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement true mmap-backed storage

**Files:**
- Modify: `vgl/storage/mmap.py`
- Modify: `tests/storage/test_mmap_store.py`

**Step 1: Write the failing test**

Use the test from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/storage/test_mmap_store.py -k sidecar`
Expected: FAIL

**Step 3: Write minimal implementation**

Write raw tensor bytes plus metadata sidecar on save, open metadata-backed arrays through `numpy.memmap(...)`, and preserve fetch/shape/dtype behavior.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/storage/test_mmap_store.py -k sidecar`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Add legacy-compatibility regression coverage

**Files:**
- Modify: `tests/storage/test_mmap_store.py`

**Step 1: Write the failing test**

Add a test proving `MmapTensorStore` can still read a legacy tensor file written with `torch.save(...)` and no metadata sidecar.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/storage/test_mmap_store.py -k legacy`
Expected: FAIL if the new implementation assumes metadata-only reads.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/storage/test_mmap_store.py -k legacy`
Expected: FAIL on missing compatibility handling.

**Step 5: Commit**

Do not commit yet.

### Task 4: Implement legacy fallback behavior

**Files:**
- Modify: `vgl/storage/mmap.py`
- Modify: `tests/storage/test_mmap_store.py`

**Step 1: Write the failing test**

Use the test from Task 3.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/storage/test_mmap_store.py -k legacy`
Expected: FAIL

**Step 3: Write minimal implementation**

When no metadata sidecar exists, fall back to loading legacy `torch.save(...)` tensor files through the previous code path.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/storage/test_mmap_store.py -k legacy`
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

No code test. Review storage docs for places that still describe mmap-backed storage as a simple file wrapper rather than a large-tensor substrate.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/storage/test_mmap_store.py`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document that `MmapTensorStore` now persists raw tensor buffers plus metadata sidecars for memory-mapped feature access, while retaining legacy file compatibility.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: expand mmap tensor store"`
