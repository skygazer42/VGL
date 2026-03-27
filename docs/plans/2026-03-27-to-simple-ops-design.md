# To Simple Graph Transform Design

## Context

`vgl.ops` now covers a broad structural surface:

- self-loop transforms through `add_self_loops(...)` and `remove_self_loops(...)`
- direction rewrites through `to_bidirected(...)` and `reverse(...)`
- relation-local subgraph extraction, k-hop extraction, compaction, and block conversion

One obvious DGL-class graph-ops gap still remains on that floor: there is no public `to_simple(...)` transform for collapsing parallel edges into one structural edge per endpoint pair.

That matters because parallel-edge deduplication is a common preprocessing and inspection step. Without a first-class transform, callers have to rebuild graphs manually whenever they want a simple graph view for debugging, statistics, sparse export, or downstream code that assumes one visible edge per pair.

## Scope Choice

Three slices were considered:

1. Implement a wide DGL-style `to_simple(...)` surface including writeback mappings, feature aggregators, and multiple output modes.
2. Add one bounded structural transform that deduplicates edges in stable order, optionally records multiplicity counts, and keeps the rest of the graph API unchanged.
3. Skip graph-level `to_simple(...)` and push users toward external NetworkX or SciPy deduplication paths.

Option 2 is the right slice.

It closes the graph-ops gap directly, stays additive, and avoids widening this batch into a second edge-feature reduction API. Option 1 is too wide for one batch, and option 3 leaves a clear public-surface hole in VGL itself.

## Recommended API

Add one public transform:

- `to_simple(graph, *, edge_type=None, count_attr=None)`

Add one matching `Graph` method:

- `graph.to_simple(...)`

Rules:

- deduplicate only the selected relation when `edge_type` is provided
- otherwise use the default or single relation, matching the current graph-op convention
- preserve current node stores, node counts, graph-store context, and feature-store context
- when `count_attr` is not `None`, attach a per-edge multiplicity tensor with that feature name

## Semantics

### Structural Order

The simplified edge set should keep stable first-occurrence order.

Concretely:

- the first time an endpoint pair `(src, dst)` appears, it becomes the visible edge
- later parallel edges with the same pair are folded into that visible edge
- self-loops stay valid and also deduplicate by exact pair

This keeps behavior deterministic and easy to reason about.

### Edge Data

This batch should keep edge-feature semantics intentionally narrow:

- edge-aligned tensor features are preserved from the first visible edge for each deduplicated pair
- non edge-aligned metadata values are copied through unchanged
- `edata["e_id"]` is dropped on the simplified relation because one simplified edge no longer corresponds one-to-one with one original public edge id
- if `count_attr` is provided, multiplicity counts are written as a `torch.long` tensor in simplified-edge order

That keeps the transform useful without inventing a broader aggregation contract.

### Heterogeneous Relations And Storage Context

`to_simple(...)` should work for homogeneous and heterogeneous relations.

For heterogeneous graphs:

- only the selected relation is simplified
- other relations remain unchanged
- node spaces remain unchanged for all participating node types

For featureless storage-backed graphs:

- declared node counts must still flow through unchanged after simplification
- adjacency shape should therefore continue to reflect the original declared node space

## Implementation Approach

The transform can stay local to `vgl.ops.structure`:

1. resolve the target relation
2. walk its `edge_index` in visible order and record the first representative position for each `(src, dst)` pair
3. build a deduplicated `edge_index` from those representative positions
4. slice edge-aligned tensor features at the representative positions, explicitly dropping `e_id`
5. optionally add the multiplicity count tensor
6. rebuild the returned graph with updated edge-feature schema while preserving node stores and graph/storage context

This keeps the batch additive and avoids touching sparse caches, sampling, or compat adapters.

## Non-Goals

- DGL-exact writeback mappings from original edges to simplified edges
- edge-feature aggregation beyond first-edge retention plus optional multiplicity counts
- mutation APIs such as in-place edge deletion
- automatic simplification across every relation in a heterogeneous graph at once
- sparse-cache deduplication or on-disk structure rewriting

## Testing Strategy

Add focused coverage for:

- homogeneous simplification collapsing parallel edges in stable order
- optional count features reflecting multiplicity
- first-edge retention for edge-aligned tensor features
- relation-local heterogeneous simplification leaving other relations untouched
- `Graph.to_simple(...)` and `vgl.ops.to_simple` public-surface exposure
- featureless storage-backed graphs preserving declared adjacency shape after simplification

Then run focused structure/core/package tests and the full repository suite.
