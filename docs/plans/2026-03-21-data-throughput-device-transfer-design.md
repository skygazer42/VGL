# Data Throughput And Device Transfer Design

**Date:** 2026-03-21
**Status:** Approved for planning

## Goal

Extend `vgl` with a first-pass data throughput layer and explicit device transfer flow that improves minibatch loading ergonomics without changing task contracts or introducing a second trainer family.

## Scope Decisions

- Focus on data loading throughput and device movement, not distributed training
- Keep the existing `Loader` and `Trainer` public entrypoints
- Add worker-backed loading only for map-style datasets
- Add explicit `Trainer(device=...)` control instead of inferring target devices from ambient model state
- Add object-level `.to()` and `pin_memory()` support for VGL graph and batch types
- Preserve current node, graph, link, and temporal task contracts
- Avoid a generalized batch reflection system for arbitrary user-defined Python objects

## Chosen Direction

Three directions were considered:

1. Keep device movement private to `Trainer` and add a minimal worker-backed loader path
2. Add object-level transfer methods across VGL graph and batch types, and let `Trainer` orchestrate them
3. Introduce a generic transfer protocol layer before shipping throughput improvements

The chosen direction is:

> Add object-level `.to()` and `pin_memory()` support for VGL graph and batch types, keep `Trainer` responsible for orchestration, and add a worker-backed `Loader` path only for map-style datasets.

This keeps the public surface coherent and reusable. A private-only transfer layer would quickly leak into other lifecycle paths such as evaluation and inference, while a generic protocol layer would be over-designed for the current codebase.

## Architecture

The public loading and training surface should stay centered on the existing entrypoints:

```python
loader = DataLoader(
    dataset=dataset,
    sampler=sampler,
    batch_size=64,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)

trainer = Trainer(
    model=model,
    task=task,
    optimizer=torch.optim.Adam,
    lr=1e-3,
    max_epochs=10,
    device="cuda",
    move_batch_to_device=True,
    non_blocking=None,
)
```

The core execution path should become:

`dataset item -> sampler.sample(...) -> batch assembly -> optional batch pinning -> optional trainer-managed device move -> model(batch)`

No new loader family and no new trainer family should be introduced.

## Public API Shape

### Loader

`Loader` should grow the following optional constructor arguments:

- `num_workers=0`
- `pin_memory=False`
- `prefetch_factor=None`
- `persistent_workers=False`

Behavioral rules:

- `num_workers=0` keeps the current single-process path
- `num_workers>0` enables an internal worker-backed path for map-style datasets only
- `prefetch_factor` is only valid when `num_workers>0`
- `persistent_workers=True` is only valid when `num_workers>0`
- `pin_memory=True` applies after VGL batch assembly, not before

### Trainer

`Trainer` should grow the following optional constructor arguments:

- `device=None`
- `move_batch_to_device=True`
- `non_blocking=None`

Behavioral rules:

- `device=None` preserves current behavior exactly
- when `device` is set, the trainer moves the model to that device during initialization
- when `move_batch_to_device=True`, every batch entering train, validation, test, sanity validation, and mid-epoch validation is moved through one shared preparation path
- `non_blocking=None` should auto-resolve to `True` only when the destination is CUDA and the batch is pinned; otherwise it should resolve to `False`

## Object Transfer Semantics

The implementation should add transfer methods to VGL-owned data structures:

- `NodeStore.to()` and `NodeStore.pin_memory()`
- `EdgeStore.to()` and `EdgeStore.pin_memory()`
- `Graph.to()` and `Graph.pin_memory()`
- `GraphView.to()` and `GraphView.pin_memory()`
- `GraphBatch.to()` and `GraphBatch.pin_memory()`
- `NodeBatch.to()` and `NodeBatch.pin_memory()`
- `LinkPredictionBatch.to()` and `LinkPredictionBatch.pin_memory()`
- `TemporalEventBatch.to()` and `TemporalEventBatch.pin_memory()`

The semantics should be:

- non-mutating: return a new object instead of changing the original in place
- tensor-aware: only tensor fields move or pin
- metadata-preserving: Python metadata and schema objects remain unchanged
- structure-preserving: hetero and temporal graph structure should remain intact after transfer

`GraphView` should transfer only the view-visible nodes and edges. The `base` reference remains provenance data and is not required to implement deep transfer semantics in this phase.

## Loader Data Flow

The loader should support two execution modes.

### Single-process mode

When `num_workers=0`, retain the current logic:

1. iterate dataset items
2. call `sampler.sample(item)`
3. expand list or tuple sample outputs
4. count batches by source seed items
5. assemble the final VGL batch with `_build_batch()`
6. optionally call `pin_memory()` on the assembled batch

This preserves current behavior exactly.

### Worker-backed mode

When `num_workers>0`, add a private worker-backed path with these constraints:

1. require a map-style dataset implementing `__len__` and `__getitem__`
2. wrap dataset and sampler in a private sampled dataset abstraction
3. let worker processes run `sampler.sample(dataset[idx])`
4. have the main process receive one sampled result per source seed item
5. flatten those per-item sampled results in the main process
6. assemble the final VGL batch with the existing `_build_batch()`
7. optionally call `pin_memory()` on the assembled batch in the main process

This is intentionally conservative. Worker processes parallelize sampling, but final VGL batch assembly remains centralized and deterministic.

## Supported Transfer Types

Automatic trainer-managed batch movement in this phase should support:

- `torch.Tensor`
- `dict`
- `list`
- `tuple`
- VGL graph and batch objects

Automatic movement should explicitly not support:

- arbitrary user-defined Python objects with ad hoc tensor attributes

When `Trainer(device=...)` and `move_batch_to_device=True` receive an unsupported batch type, the trainer should fail early with a clear `TypeError`.

## Compatibility Constraints

Phase 1 of this work should preserve two important compatibility points:

- `GraphBatch.graphs` stays in place and remains usable by current examples and tests
- `Loader` does not expose a second `collate_fn` contract; `_build_batch()` remains the single VGL batching contract

This avoids rewriting current graph-classification examples and keeps sampler-to-batch semantics centralized.

## Error Handling

The feature should fail early for invalid configuration:

- `Loader(num_workers>0)` with a non-map-style dataset raises `TypeError`
- `prefetch_factor` with `num_workers=0` raises `ValueError`
- `persistent_workers=True` with `num_workers=0` raises `ValueError`
- `Trainer(device=...)` with unsupported batch types raises `TypeError`
- `Trainer(precision="fp16-mixed")` with a non-CUDA target device raises the current `ValueError`
- sharpness-aware optimizers with a grad scaler remain an invalid combination

Non-tensor fields encountered during `pin_memory()` should be ignored silently rather than treated as errors.

## Testing Strategy

The test plan should cover four layers.

### Graph and store transfer tests

- `NodeStore.to()` and `pin_memory()`
- `EdgeStore.to()` and `pin_memory()`
- `Graph.to()` and `pin_memory()`
- `GraphView.to()` and `pin_memory()`
- homo, hetero, and temporal graph cases
- non-mutating behavior

### Batch transfer tests

- `GraphBatch.to()` and `pin_memory()`
- `NodeBatch.to()` and `pin_memory()`
- `LinkPredictionBatch.to()` and `pin_memory()`
- `TemporalEventBatch.to()` and `pin_memory()`
- graph tensors, index tensors, labels, timestamps, and event features all move correctly
- metadata remains unchanged

### Loader throughput tests

- `num_workers=0` preserves current semantics
- `num_workers>0` behaves equivalently on `ListDataset`
- `pin_memory=True` produces pinned tensor-backed batches where supported
- invalid worker configuration fails early

### Trainer device transfer tests

- `device="cpu"` moves model and VGL batches consistently
- `move_batch_to_device=False` moves only the model
- train, validation, test, sanity validation, and mid-epoch validation all use the same batch preparation path
- mixed precision behavior remains correct with explicit devices
- unsupported batch types fail early with a clear error

## Repository Touchpoints

This work should primarily touch:

- `vgl/dataloading/loader.py`
- `vgl/engine/trainer.py`
- `vgl/graph/stores.py`
- `vgl/graph/graph.py`
- `vgl/graph/view.py`
- `vgl/graph/batch.py`
- `tests/data/`
- `tests/core/`
- `tests/train/`
- documentation and examples that need to show the new knobs

## Explicit Non-Goals

Do not include in this phase:

- distributed training
- DDP, FSDP, or multi-device strategies
- arbitrary custom-object auto-transfer by reflection
- a new loader class or alternate collate API
- performance benchmarking infrastructure
- deep profiler integrations

These belong to later phases and should not distort the first stable throughput and transfer contract.

## Acceptance Criteria

This design is complete when:

1. `Loader` supports optional worker-backed sampling for map-style datasets
2. VGL graph and batch types implement stable `.to()` and `pin_memory()` behavior
3. `Trainer(device=...)` provides explicit model and batch placement without changing current default behavior
4. train, validation, test, sanity validation, and mid-epoch validation share one batch preparation path
5. current node, graph, link, and temporal flows remain green
6. documentation and tests reflect the new public controls

## Next Step

The next step is to write the implementation plan with exact files, tests, commands, and commit checkpoints.
