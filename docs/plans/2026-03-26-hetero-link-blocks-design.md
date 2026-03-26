# Heterogeneous Link Block Output Design

## Context

VGL already has the right low-level pieces for relation-local heterogeneous block output:

- heterogeneous link sampling can already materialize one stitched or local hetero sampled graph
- `to_block(...)` already builds relation-local heterogeneous `Block` objects when an `edge_type` is selected
- `LinkPredictionRecord.blocks` and `LinkPredictionBatch.blocks` already batch per-layer block lists

What is still missing is the bridge between those pieces. Today `LinkNeighborSampler(..., output_blocks=True)` still rejects heterogeneous graphs, including the coordinator-backed stitched heterogeneous path that already reconstructs the right sampled frontier subgraph for large-graph training.

That leaves a visible DGL-style training gap. Users can sample heterogeneous link frontiers, and they can build relation-local blocks manually, but they cannot ask the sampler to emit layer-wise blocks directly from the same sampled message-passing view.

## Scope Choice

Three slices were considered:

1. Add full multi-relation heterogeneous block lists for every sampled edge type.
2. Add relation-local heterogeneous block output for the supervised `edge_type` only.
3. Add heterogeneous node and link block output together.

Option 2 is the right slice.

The existing `Block` abstraction is already explicitly relation-local. Extending it to represent one block per relation per hop would widen the public contract and immediately spill into batching semantics for mixed-edge-type records. By keeping the output tied to the supervised relation only, VGL can unlock the high-value DGL-style link-training workflow without inventing a richer multi-relation block container first.

## Recommended Semantics

Keep the public API unchanged:

- `LinkNeighborSampler(..., output_blocks=False)`
- `LinkNeighborSampler(..., output_blocks=True)`
- `LinkPredictionRecord.blocks`
- `LinkPredictionBatch.blocks`

New behavior:

- local homogeneous behavior stays unchanged
- local heterogeneous link sampling may now emit blocks when every sampled record in the item shares one `edge_type`
- stitched heterogeneous link sampling through `LocalSamplingCoordinator` may now emit blocks under the same single-relation rule
- each emitted block remains relation-local and is built with `to_block(sampled_graph, ..., edge_type=record.edge_type)`
- blocks stay ordered outer-to-inner

Still unsupported in this batch:

- multi-relation block lists
- `output_blocks=True` when one sampled item mixes multiple heterogeneous `edge_type` values
- heterogeneous node block output
- temporal link block output

## Construction Strategy

The crucial design choice is to build blocks from the final sampled message-passing graph, not from raw frontier-expansion state.

That matches the current homogeneous path and preserves two important properties:

1. feature overlays fetched after sampling remain visible on block graphs
2. seed-edge exclusion still removes positive supervision edges from block message passing

The only new state the executor/sampler must retain is hop history in a type-aware form. For heterogeneous link sampling, the relevant block destination frontier is the cumulative node set for the supervised relation's destination node type. That means the implementation can:

1. retain cumulative hop snapshots by node type during heterogeneous expansion
2. pick the supervised relation's destination-type snapshots during materialization
3. map those snapshots onto the sampled graph's local `n_id` ordering
4. call `to_block(..., edge_type=supervised_edge_type)` for each outer-to-inner layer

This works for both local and stitched heterogeneous paths because both already produce sampled graphs with stable per-type `n_id`.

## Mixed Edge-Type Handling

`LinkPredictionBatch.blocks` currently assumes one block schema per layer. That is already true for homogeneous blocks, but it becomes explicit for heterogeneous relation-local blocks.

This batch should therefore fail clearly when block-bearing records with different `edge_type` values are batched together. That keeps failure deterministic and honest instead of letting `_batch_block_layer(...)` blow up later with a lower-level schema mismatch.

## Testing Strategy

Add focused regressions for:

- local heterogeneous `LinkNeighborSampler(..., output_blocks=True)` returning relation-local blocks
- stitched heterogeneous `LinkNeighborSampler(..., output_blocks=True)` returning relation-local blocks through `LocalSamplingCoordinator`
- feature overlays remaining visible on heterogeneous stitched block graphs
- fixed block count across heterogeneous hop lists even when the destination-type frontier stops growing
- mixed-edge-type block batches failing clearly

Then rerun the focused heterogeneous link/block suites and the full repository suite before merge.
