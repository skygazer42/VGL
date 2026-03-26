# Storage-Backed Graph Node Count Design

## Context

VGL already supports storage-backed graphs through `Graph.from_storage(...)`, lazy node and edge feature loading, adjacency caching, loader plan execution, and DGL export. One bottom-layer gap still remains in the graph abstraction itself: node cardinality inference is still feature-driven.

`Graph._node_count(...)` currently inspects loaded node tensors and otherwise raises. That works for in-memory graphs and for storage-backed graphs that expose at least one node-aligned feature tensor, but it breaks as soon as a storage-backed graph is featureless or isolate-heavy. In those cases the graph store still knows the node counts, yet basic graph operations such as `adjacency(...)`, `to_dgl()`, and `to_block(...)` either lose isolated nodes or fail outright.

## Scope Choice

This batch should not redesign lazy slicing or graph-store-backed subgraph materialization. It should only make graph-level node cardinality aware of graph-store metadata that already exists.

1. Extend `GraphStore` with `num_nodes(node_type)`.
2. Retain `graph_store` on `Graph.from_storage(...)`.
3. Teach `Graph._node_count(...)` and `GraphView._node_count(...)` to consult `graph_store` when node tensors do not determine count.
4. Preserve that context through `Graph.to(...)` / `pin_memory()`.

That is enough to unblock featureless storage-backed graphs for the existing base operations that only need counts.

## Recommended Behavior

### Graph store contract

- `GraphStore.num_nodes(node_type)` returns the declared node cardinality for a node type
- `InMemoryGraphStore` should implement it from its existing internal `num_nodes` mapping
- local distributed adapters that wrap a `GraphStore` should forward it too

### Graph count resolution

For `Graph` and `GraphView`:

1. If a node store contains a rank-1-or-higher aligned tensor, use its leading dimension as today.
2. Otherwise, if `graph_store` is present, use `graph_store.num_nodes(node_type)`.
3. Otherwise, preserve the current `ValueError`.

### Scope win

This should make these paths work on featureless storage-backed graphs without inventing fake node features:

- `graph.adjacency(...)`
- `graph.to_dgl()`
- `graph.to_block(...)`
- any other path that already depends only on node counts plus structure

## Deferred Scope

This batch should not attempt:

- graph-store-backed lazy slicing for subgraph outputs
- preserving `graph_store` on newly materialized derived graphs such as `node_subgraph(...)`
- distributed remote graph-store APIs beyond forwarding `num_nodes(...)` through local adapters
- any widening of public graph batch or block contracts

## Testing Strategy

Add focused regressions for:

- featureless homogeneous storage-backed graphs preserving declared adjacency shape and graph-store context
- featureless heterogeneous storage-backed graphs preserving per-type adjacency shape
- featureless homogeneous storage-backed graphs exporting to DGL with the declared `num_nodes()` intact
- `to_block(...)` working on a featureless storage-backed homogeneous graph using graph-store counts for destination validation

Then rerun focused storage/DGL tests and the full repository suite before merge.
