# Multi-Relation Heterogeneous Sampler Block Output Design

## Context

VGL now has the graph-op primitive needed for full heterogeneous message-flow layers:

- relation-local `Block`
- multi-relation `HeteroBlock`
- `to_hetero_block(...)` for one heterogeneous bipartite layer

The remaining gap is sampler integration. `NodeNeighborSampler(..., output_blocks=True)` and `LinkNeighborSampler(..., output_blocks=True)` still materialize heterogeneous blocks through the older relation-local path only. That leaves several DGL-style training workloads unsupported or artificially constrained:

- heterogeneous node sampling with multiple inbound relations into the supervised frontier
- heterogeneous node sampling where later hops introduce additional destination node types
- heterogeneous link sampling where one sampled item contains multiple `edge_type` values
- coordinator-backed stitched variants of the same workloads

## Scope Choice

Three contract options were considered:

1. Add parallel `hetero_blocks` fields and keep `blocks` relation-local.
2. Replace sampler `blocks` with a widened union: `Block | HeteroBlock`.
3. Keep the current relation-local contract and only improve graph ops.

Option 2 is the right slice.

Adding a second field would duplicate one concept, force callers to branch earlier, and make batching more awkward. Stopping at graph ops would preserve the current training gap. Widening `blocks` keeps one layer-sequence abstraction while letting homogeneous paths remain unchanged and heterogeneous paths graduate to the richer container when needed.

## Recommended Semantics

Keep homogeneous behavior unchanged:

- homogeneous samplers still emit `list[Block]`

Change heterogeneous sampler behavior:

- heterogeneous node sampling with `output_blocks=True` emits one block layer per hop using the full heterogeneous frontier state
- heterogeneous link sampling with `output_blocks=True` does the same, even when the sampled records mix `edge_type` values
- each heterogeneous layer is a `HeteroBlock`
- the layer order stays outer-to-inner

This removes the current heterogeneous sampler restrictions:

- no more “exactly one inbound edge_type” requirement for node block output
- no more “single edge_type” requirement for heterogeneous link block output

The supervision contract does not change:

- `NodeBatch.seed_index` stays flat
- `LinkPredictionBatch.edge_type` / `edge_types` / `edge_type_index` stay as they are today
- `SampleRecord.blocks`, `NodeBatch.blocks`, `LinkPredictionRecord.blocks`, and `LinkPredictionBatch.blocks` simply widen to hold `Block | HeteroBlock`

## Construction Strategy

For heterogeneous sampled graphs, each block layer should be built from the final sampled message-passing graph plus the cumulative per-type hop snapshots already tracked during expansion.

For one hop snapshot:

1. Map each hop snapshot’s public ids back onto the sampled subgraph through per-type `n_id`.
2. Call `to_hetero_block(sampled_graph, dst_nodes_by_type=hop_snapshot, edge_types=tuple(sampled_graph.edges))`.
3. Repeat for every outer-to-inner hop.

This keeps the same important invariant as the homogeneous path:

- block graphs are derived from the final sampled graph, so fetched feature overlays and seed-edge exclusion stay visible

## Graph-Op Stability Requirement

`to_hetero_block(...)` needs one small hardening pass before sampler integration:

- selected relations with empty destination frontiers must still materialize valid empty node/edge stores
- source-store bookkeeping must stay schema-stable even when one relation contributes zero predecessors in a specific layer

Without that, heterogeneous block batching across samples and hops is too fragile because graph schemas can vary with frontier exhaustion.

## Batching Semantics

`NodeBatch` and `LinkPredictionBatch` already batch block layers hop-by-hop. That logic should extend as follows:

- a block layer may be either all `Block` or all `HeteroBlock`
- relation-local homogeneous behavior stays unchanged
- heterogeneous sampler-produced layers batch through a new `HeteroBlock` path that concatenates per-type `src_n_id` / `dst_n_id` and batches the wrapped heterogeneous graphs
- mixed `Block` and `HeteroBlock` objects in the same layer should fail clearly

## Testing Strategy

Add focused regressions for:

- local heterogeneous node sampling with multiple relations producing `HeteroBlock` layers
- stitched heterogeneous node sampling with multiple relations producing `HeteroBlock` layers
- local heterogeneous link sampling with mixed `edge_type` supervision producing `HeteroBlock` layers
- stitched heterogeneous link sampling with mixed `edge_type` supervision producing `HeteroBlock` layers
- `NodeBatch.from_samples(...)` and `LinkPredictionBatch.from_records(...)` batching `HeteroBlock` layers
- `NodeBatch.to(...)`, `LinkPredictionBatch.to(...)`, and `pin_memory()` handling `HeteroBlock`
- `to_hetero_block(...)` staying schema-stable with empty frontiers

Then rerun the focused node/link/batch/block suites and the full repository suite before merge.
