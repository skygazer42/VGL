# Mmap Tensor Store Expansion Design

## Goal

Upgrade `vgl.storage.MmapTensorStore` from a lazy `torch.load(...)` wrapper into a true memory-mapped tensor store suitable for large tensor-backed graph features.

## Why This Batch

The current `MmapTensorStore` name promises large-scale storage behavior, but the implementation still loads the full tensor file into memory on first access. That undercuts one of the most important remaining substrate goals for large-graph training: storage layers that avoid eagerly materializing full feature tensors. This is a better next step than adding more high-level APIs because it strengthens a low-level primitive that other feature-store-backed flows can adopt later.

## Recommended Approach

Keep the public class name and `save(...)` / `fetch(...)` interface stable, but change the on-disk representation for newly written stores:

- the provided path becomes the raw contiguous tensor buffer
- a small JSON sidecar at `<path>.meta.json` records shape and dtype
- reads use `numpy.memmap(...)` plus `torch.from_numpy(...)` to expose tensor slices without eagerly loading the full array

For compatibility, `MmapTensorStore` should still detect legacy `torch.save(...)` payloads when the metadata sidecar is absent and fall back to the old `torch.load(...)` behavior.

## Constraints

- Do not redesign the `TensorStore` protocol.
- Do not add write-in-place mutation APIs in this batch.
- Do not require callers to change how they construct `MmapTensorStore(path)` or call `save(path, tensor)`.
- Preserve existing fetch semantics and dtype reporting.

## Testing Strategy

Use TDD in `tests/storage/test_mmap_store.py`:

1. Add a failing regression test proving `save(...)` now emits metadata sidecar files while preserving fetch, shape, and dtype behavior.
2. Add a failing regression test proving `MmapTensorStore` can still read legacy `torch.save(...)` tensor files without metadata.
3. Run the full suite after implementation because this is a shared storage substrate.

## Documentation Impact

Update `README.md`, `docs/core-concepts.md`, and `docs/quickstart.md` so the storage foundation is described as including true mmap-backed tensor access for large feature tables instead of just in-memory stores.
