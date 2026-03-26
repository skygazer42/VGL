# Inbound And Outbound Frontier Subgraph Ops Design

## Context

`vgl.ops` already covers induced node/edge subgraphs, k-hop extraction, compaction, line graphs, metapath reachability, walks, and block rewrites. One DGL-class graph-ops gap still remains at the frontier-subgraph layer: there is no direct equivalent of `in_subgraph(...)` or `out_subgraph(...)`.

Those operators matter because they fill the semantic gap between edge filtering and neighborhood expansion. They let callers slice one graph down to just the incoming or outgoing edges of a frontier without compacting node ids, which is useful for message-flow inspection, sampler debugging, and interoperability with DGL-style graph preprocessing.

## Scope Choice

Three plausible next slices were considered:

1. Add only relation-local frontier extraction for one selected edge type.
2. Add DGL-style `in_subgraph(...)` and `out_subgraph(...)` that preserve node space and compose across all relations.
3. Jump to graph mutation APIs such as `add_edges(...)` and `remove_edges(...)`.

Option 2 is the right batch. It directly closes a graph-ops parity gap, stays additive, and is smaller than graph mutation or multi-relation block redesign. It also complements the current `node_subgraph(...)` and `edge_subgraph(...)` surface cleanly: induced subgraphs compact or preserve selected structure, while frontier subgraphs keep the original node space and only filter edges by source or destination frontier.

## Recommended Design

Add two new public ops:

- `in_subgraph(graph, nodes)`
- `out_subgraph(graph, nodes)`

Both should follow DGL's default semantics closely:

- preserve the original node space instead of relabeling or compacting
- preserve the original relation set for heterogeneous graphs
- filter edges by destination frontier for `in_subgraph(...)`
- filter edges by source frontier for `out_subgraph(...)`
- always store the extracted raw/public edge ids as `edata["e_id"]`

For homogeneous graphs, `nodes` may be any rank-1 node-id collection accepted by existing ops. For heterogeneous graphs, `nodes` should be a dictionary keyed by node type. Each relation independently reads the relevant source or destination node-type frontier and extracts matching edges, then all relation-local outputs are composed back into one graph with the same schema as the input.

## Storage And Feature Semantics

The returned graph should keep the original node stores instead of eagerly rebuilding them. That preserves lazy node features for storage-backed graphs and matches the intended DGL-style lazy feature copy behavior closely enough for this batch.

Edge stores, however, should be rebuilt with sliced edge-aligned tensors because the structure really changes. For graphs that already expose public edge ids through `edata["e_id"]`, the new subgraph should preserve those ids. Otherwise it should synthesize `e_id` from the selected raw edge positions.

Because `in_subgraph(...)` and `out_subgraph(...)` preserve the original node space, they can safely retain `feature_store` and `graph_store` on the returned graph. That is important for featureless storage-backed graphs: node counts stay valid even when node tensors are absent, and node-aligned lazy features remain discoverable from the same retained storage context.

## API And Error Handling

`Graph` should grow two convenience methods:

- `graph.in_subgraph(nodes)`
- `graph.out_subgraph(nodes)`

Input validation should stay explicit:

- heterogeneous graphs with more than one node type require `nodes` keyed by node type
- unknown node-type keys should raise `ValueError`
- duplicate input ids may be normalized to stable unique frontiers, matching the existing helper behavior in `vgl.ops.subgraph`

This batch should not add DGL's optional `relabel_nodes`, `store_ids`, or `output_device` arguments. VGL can expose those knobs later if real pressure shows up. For now the contract should stay small and opinionated: preserved node space plus `e_id` always on.

## Testing Strategy

Add focused coverage in `tests/ops/test_subgraph_ops.py` for:

- homogeneous inbound frontier extraction
- homogeneous outbound frontier extraction
- heterogeneous composed frontier extraction across multiple relations
- `e_id` preservation or synthesis
- validation on ambiguous heterogeneous tensor input

Add graph-API bridge checks in `tests/core/test_graph_ops_api.py`, export assertions in `tests/test_package_layout.py`, and one storage-backed regression in `tests/core/test_feature_backed_graph.py` proving featureless storage-backed frontier subgraphs still preserve node-count-sensitive adjacency shape.

## Non-Goals

This batch should not include:

- node relabeling or compaction flags on frontier subgraphs
- graph mutation APIs such as `add_edges(...)` or `remove_edges(...)`
- multi-relation `Block` adapters
- block-graph subgraph extraction
- k-hop frontier extraction beyond the existing `khop_*` operators
- distributed coordinator-specific frontier subgraph helpers
