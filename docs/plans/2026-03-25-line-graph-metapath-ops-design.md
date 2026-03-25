# Line Graph And Metapath Ops Design

## Context

`vgl.ops` already covers self-loop rewrites, bidirection conversion, induced subgraphs, k-hop extraction, and compaction, but it still lacks two high-signal DGL-class structure transforms:

- `line_graph(...)` for edge-centric message-passing and higher-order topology workflows
- `metapath_reachable_graph(...)` for heterogeneous path composition and metapath-based preprocessing

Those are substrate-level graph operations, not model-specific features, so they fit the current goal of widening the graph-ops surface without touching loaders, samplers, or trainer behavior.

## Scope Choice

Three plausible batches were considered:

1. Add only `line_graph(...)` for homogeneous graphs.
2. Add `line_graph(...)` plus `metapath_reachable_graph(...)` as one compact topology batch.
3. Jump to walk-style operations such as random walk or node2vec sampling.

Option 2 is the right slice. It adds one edge-centric transform and one hetero path transform, covers both homogeneous and heterogeneous graph-ops gaps, and stays small enough to test directly in `vgl.ops` and the `Graph` convenience API.

## Recommended Design

Add a new `vgl.ops.path` module and export:

- `line_graph(graph, *, edge_type=None, backtracking=True, copy_edata=True)`
- `metapath_reachable_graph(graph, metapath, *, relation_name=None)`

`line_graph(...)` should support graphs with exactly one selected relation. Each original edge becomes one node in the output graph. Two line-graph nodes connect when the destination of the first edge equals the source of the second edge. When `backtracking=False`, immediate reversals such as `(u, v)` followed by `(v, u)` are excluded. The output graph is homogeneous with node count equal to the selected edge count. When `copy_edata=True`, edge-aligned tensors from the source relation become node-aligned tensors on the line graph, and the line-graph node store should also carry `n_id` equal to the original edge ids.

`metapath_reachable_graph(...)` should support heterogeneous graphs and single-node-type multi-relation graphs. The input `metapath` is an ordered sequence of edge types whose adjacent endpoint types must line up. The output graph keeps the original node stores for the metapath start and end node types and materializes one derived relation whose edge index contains the deduplicated reachable `(src, dst)` pairs implied by the metapath. This batch should not count multiplicity, emit path instances, or propagate intermediate-edge features.

## API And Semantics

`Graph` should grow two convenience bridges:

- `graph.line_graph(...)`
- `graph.metapath_reachable_graph(...)`

`line_graph(...)` should fail early when the chosen relation is not homogeneous, because the initial implementation intentionally models edge adjacency over one node space. `metapath_reachable_graph(...)` should fail early for an empty metapath or for edge-type chains whose node types do not compose.

Derived relation naming should be deterministic. The recommended default is:

- line graph edge type: `("node", "line", "node")`
- metapath relation name: `"__".join(rel for _, rel, _ in metapath)`

The output of `metapath_reachable_graph(...)` should be a hetero graph with exactly one edge type, even when the start and end node types match. That avoids special-case behavior for non-`"node"` self-type relations and keeps the operation stable across heterogeneous and multi-relation inputs.

## Testing Strategy

Add focused unit coverage in `tests/ops/` for:

- homogeneous `line_graph(...)`
- `backtracking=False` pruning
- copied edge features becoming line-graph node features
- heterogeneous `metapath_reachable_graph(...)`
- single-node-type multi-relation metapaths
- invalid metapath chaining errors

Then extend `tests/core/test_graph_ops_api.py` to prove the `Graph` method bridges call through correctly, and update `tests/test_package_layout.py` so the new ops are part of the stable exported surface.

## Non-Goals

This batch should not include:

- random-walk or walk-sampling APIs
- path-count weights or multiplicity-aware metapath outputs
- heterogeneous line graphs
- feature propagation from intermediate metapath edges into the derived graph
- loader, sampler, or distributed-runtime integration
