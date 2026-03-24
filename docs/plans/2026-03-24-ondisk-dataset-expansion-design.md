# On-Disk Dataset Expansion Design

## Goal

Expand `vgl.data.OnDiskGraphDataset` so it can round-trip heterogeneous and temporal graphs, not just homogeneous non-temporal graphs.

## Why This Batch

`OnDiskGraphDataset` is part of the current data-ecosystem foundation, but its serializer still raises on any graph that is not homogeneous and non-temporal. That is now one of the clearest remaining substrate gaps relative to DGL-style graph tooling: the data layer should not drop support for graph kinds that the in-memory core already handles well. This batch is high leverage because it strengthens dataset portability without changing model code or loader contracts.

## Recommended Approach

Keep the existing dataset container shape intact: one `manifest.json` plus one `graphs.pt` payload. Generalize the per-graph payload format rather than replacing the storage container. Each graph payload should record:

- `nodes`: a mapping from node type to its feature dictionary
- `edges`: a mapping from edge type tuple to its edge-data dictionary, including `edge_index`
- `time_attr`: optional temporal edge-field name

With that payload, `deserialize_graph(...)` can reconstruct the graph through `Graph.hetero(...)` for non-temporal graphs and `Graph.temporal(...)` when `time_attr` is present. Homogeneous graphs still serialize and deserialize through the same generalized path, so the public `OnDiskGraphDataset` API stays unchanged while coverage broadens underneath it.

## Constraints

- Do not introduce a new file format versioning scheme in this batch.
- Do not add lazy loading, mmap-backed dataset reads, or split-aware dataset views yet.
- Do not redesign `DatasetManifest` or loader APIs.
- Preserve existing homogeneous round-trip behavior.

## Testing Strategy

Use TDD in `tests/data/test_ondisk_dataset.py`:

1. Add a failing heterogeneous round-trip test that checks typed nodes, typed edges, and edge features survive disk serialization.
2. Add a failing temporal round-trip test that checks `schema.time_attr`, `edge_index`, and temporal edge data survive disk serialization.
3. Keep the existing homogeneous regression test green.
4. Run the full suite after docs are updated, because the serializer touches a shared foundation layer.

## Documentation Impact

Refresh `README.md`, `docs/core-concepts.md`, and `docs/quickstart.md` so the data foundation is described as supporting heterogeneous and temporal on-disk graph datasets instead of only a homogeneous toy format.
