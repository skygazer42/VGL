# Heterogeneous Link Block Output Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add relation-local heterogeneous `output_blocks=True` support to `LinkNeighborSampler` for both local and stitched link sampling while keeping the existing `Block` API unchanged.

**Architecture:** Reuse the existing relation-local `to_block(...)` path instead of inventing a new block container. Teach heterogeneous link expansion to retain cumulative per-type hop snapshots, materialize blocks from the sampled message-passing graph using the supervised relation's destination node type, and reject mixed-edge-type block batches clearly.

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

### Task 1: Add failing heterogeneous link block regressions

**Files:**
- Modify: `tests/data/test_link_neighbor_sampler.py`
- Modify: `tests/integration/test_foundation_partition_local.py`
- Modify: `tests/core/test_link_prediction_batch.py`

**Step 1: Write the failing test**

Add regressions proving:

- local heterogeneous `LinkNeighborSampler(..., output_blocks=True)` returns relation-local blocks for the selected `edge_type`
- stitched heterogeneous `LinkNeighborSampler(..., output_blocks=True)` returns relation-local blocks through `LocalSamplingCoordinator`
- heterogeneous stitched block graphs keep fetched node / edge feature overlays
- fixed hop count is preserved even when later heterogeneous expansions do not add new destination-type nodes
- `LinkPredictionBatch.from_records(...)` fails clearly when records with blocks mix multiple `edge_type` values

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py -k "hetero and output_blocks" tests/integration/test_foundation_partition_local.py -k "hetero and block" tests/core/test_link_prediction_batch.py -k "mixed and blocks"`
Expected: FAIL because heterogeneous link block output is still rejected and mixed-edge block batches do not yet fail explicitly.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py -k "hetero and output_blocks" tests/integration/test_foundation_partition_local.py -k "hetero and block" tests/core/test_link_prediction_batch.py -k "mixed and blocks"`
Expected: FAIL on the current heterogeneous `output_blocks` guard or missing mixed-edge validation.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement local and stitched heterogeneous link block materialization

**Files:**
- Modify: `vgl/dataloading/sampler.py`
- Modify: `vgl/dataloading/executor.py`
- Modify: `vgl/dataloading/materialize.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py -k "hetero and output_blocks" tests/integration/test_foundation_partition_local.py -k "hetero and block"`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- heterogeneous link-hop tracking as cumulative per-type snapshots
- local heterogeneous `LinkNeighborSampler(..., output_blocks=True)` support when all sampled records share one `edge_type`
- stitched heterogeneous link block support through the coordinator path using the same per-type hop model
- relation-local block construction from the sampled message-passing graph via `to_block(..., edge_type=...)`
- fixed hop-count behavior by retaining one destination-type snapshot per configured fanout

Implementation rules:

- build blocks for the supervised relation only
- preserve current homogeneous behavior exactly
- keep seed-edge exclusion flowing through `_link_message_passing_graph(...)`
- fail clearly if one sampled item mixes heterogeneous `edge_type` values under `output_blocks=True`

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py -k "hetero and output_blocks" tests/integration/test_foundation_partition_local.py -k "hetero and block"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Tighten block batching validation

**Files:**
- Modify: `vgl/graph/batch.py`
- Modify: `tests/core/test_link_prediction_batch.py`

**Step 1: Write the failing test**

Use the mixed-edge-type block regression from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/core/test_link_prediction_batch.py -k "mixed and blocks"`
Expected: FAIL

**Step 3: Write minimal implementation**

Add one explicit `LinkPredictionBatch.from_records(...)` validation that rejects block-bearing records with multiple `edge_type` values before `_batch_blocks(...)` reaches a lower-level schema mismatch.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/core/test_link_prediction_batch.py -k "mixed and blocks"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 4: Refresh docs and run full regression

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review link-sampling docs for places that still say heterogeneous stitched link block output is unsupported or imply blocks are homogeneous-only.

**Step 2: Run test to verify the code path state**

Run: `python -m pytest -q tests/data/test_link_neighbor_sampler.py tests/integration/test_foundation_partition_local.py tests/core/test_link_prediction_batch.py`
Expected: PASS if the heterogeneous block path is implemented cleanly.

**Step 3: Write minimal implementation**

Document that:

- `LinkNeighborSampler(..., output_blocks=True)` now supports relation-local heterogeneous link blocks
- coordinator-backed stitched heterogeneous link sampling can emit the same relation-local blocks
- block lists still require one supervised relation schema per sampled batch item

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add hetero link block output"`
