# Lazy Feature-Backed Graph Design

## Goal

Make `Graph.from_storage(...)` lazy for feature tensors so storage-backed graphs stop materializing every node and edge feature table at construction time.

## Why This Batch

VGL now has stronger substrate pieces for large-graph workflows: `FeatureStore`, on-disk dataset support, distributed partition metadata, and a true `MmapTensorStore`. But one critical gap remains: `Graph.from_storage(...)` still immediately fetches every feature tensor into memory. That means storage-backed graphs only look scalable at the API boundary; the current implementation still pays the full eager materialization cost up front. Fixing this is the most direct next step toward real large-graph training because it lets storage-backed graphs delay feature IO until the specific tensors are actually touched.

## Recommended Approach

Keep the public `Graph`, `NodeStore`, and `EdgeStore` surface stable. Instead of introducing a second graph type, teach stores created from feature storage to resolve tensors lazily and cache them after first access. The core idea is:

- `NodeStore.from_feature_store(...)` should create a store with feature metadata, not eagerly fetched tensors
- `EdgeStore.from_storage(...)` should eagerly keep `edge_index`, but defer non-structural edge features
- store attribute access such as `graph.x`, `graph.y`, or `graph.edata["weight"]` should fetch exactly that feature on demand and cache it
- iteration paths like `store.data.items()` may materialize any remaining lazy features when callers explicitly walk the whole store

This preserves the current user-facing API and keeps the change local to the graph/storage boundary. It also composes well with `MmapTensorStore`: the first access can now fetch from mmap-backed storage without `Graph.from_storage(...)` pulling the entire table up front.

## Constraints

- Do not add a separate lazy graph class.
- Do not redesign `FeatureStore` or `TensorStore` protocols in this batch.
- Do not change sampler semantics or task APIs.
- Preserve existing homogeneous and heterogeneous behavior.

## Testing Strategy

Use TDD in `tests/core/test_feature_backed_graph.py` and one integration path:

1. Add a failing regression test proving `Graph.from_storage(...)` does not call `FeatureStore.fetch(...)` during construction.
2. Add a failing regression test proving node and edge features are fetched lazily and cached after first access.
3. Add an integration regression that routes a storage-backed graph built on `MmapTensorStore` through the existing loader/trainer path.
4. Run the full suite after docs are updated because this changes a shared graph substrate.

## Documentation Impact

Refresh `README.md`, `docs/core-concepts.md`, and `docs/quickstart.md` so storage-backed graphs are described as lazily resolving feature tensors instead of eagerly materializing them at construction time.
