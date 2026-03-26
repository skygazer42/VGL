# Node Neighbor Sampler Block Output Design

## Context

VGL now has a relation-local `Block` container and `to_block(...)` transform, but mini-batch node sampling still materializes only one sampled subgraph plus seed indices. Compared with DGL-style layer-wise training, the biggest remaining gap is that neighbor sampling cannot emit stacked message-flow blocks for each hop.

That gap matters more than another isolated graph transform because many neighborhood-sampled GNN training loops want one block per layer, ordered from outer frontier to inner seeds, while still preserving the sampled subgraph for feature access and compatibility with the current loader contract.

## Scope Choice

Three implementation slices were considered:

1. Change `NodeNeighborSampler` to return only blocks and remove the sampled subgraph.
2. Keep the existing `graph` and `seed_index` behavior, and optionally attach `blocks` for homogeneous node sampling.
3. Add block output for homogeneous, heterogeneous, link, temporal, and distributed node sampling in one batch.

Option 2 is the right slice. Replacing the sampled subgraph would break the existing public surface and force downstream rewrites. Option 3 pulls in a much larger heterogeneous and distributed block abstraction problem. Optional homogeneous block output gives the highest training value while keeping the runtime change narrow.

## Recommended API

Extend `NodeNeighborSampler` with:

- `NodeNeighborSampler(..., output_blocks=False)`

Extend sampled-node records and batches with:

- `SampleRecord.blocks: list[Block] | None`
- `NodeBatch.blocks: list[Block] | None`

Semantics:

- when `output_blocks=False`, behavior stays unchanged
- when `output_blocks=True` on homogeneous local sampling, each sample carries `len(num_neighbors)` blocks
- blocks are ordered outer-to-inner, matching a standard multi-layer message-passing stack
- `NodeBatch.from_samples(...)` batches each block layer independently across samples

## Construction Strategy

Blocks must be built from the already sampled subgraph, not from the original source graph. Using the original graph with only destination nodes would pull back unsampled incoming edges and violate the fanout-sampled neighborhood.

To support this, homogeneous neighbor expansion should retain cumulative node states for each hop:

- hop 0: seed nodes
- hop 1: nodes visited after the first expansion
- ...
- hop L: all nodes in the sampled subgraph

For a fanout list of length `L`, block construction should use destination sets in reverse cumulative order:

- block 0 uses destination nodes from hop `L - 1`
- ...
- block `L - 1` uses the original seed nodes

Each block is built as `to_block(sampled_subgraph, dst_nodes=...)`, where `dst_nodes` are subgraph-local ids. This keeps the block edge set aligned with the sampled subgraph.

## Id Preservation

The sampled subgraph already carries `n_id` and `e_id` that map back to the original graph. `to_block(...)` should preserve those external ids when they already exist instead of overwriting them with local subgraph indices. That lets sampled blocks expose:

- `src_n_id` and `dst_n_id` in original graph id space
- `edata["e_id"]` in original graph edge id space

This preservation is required so downstream training or bookkeeping code can relate blocks back to the global graph.

## Validation And Deferred Scope

This batch should fail clearly when `output_blocks=True` is requested for unsupported cases:

- heterogeneous node sampling
- stitched or distributed homogeneous sampling

Both need a larger cross-relation or cross-partition block representation and should not be approximated here.

This batch should not change:

- the loader return type
- link or temporal sampler outputs
- heterogeneous block abstractions
- distributed block materialization

## Testing Strategy

Add focused tests for:

- homogeneous `NodeNeighborSampler(..., output_blocks=True)` sample output
- block ordering and count stability across configured hops
- preservation of global `n_id` and `e_id` inside blocks
- ensuring blocks include only sampled edges, not full original in-edges
- `NodeBatch.from_samples(...)` batching per-layer blocks across multiple samples
- `NodeBatch.to(...)` and `NodeBatch.pin_memory()` carrying blocks
- clear errors for unsupported heterogeneous block output

Then rerun the full suite before merge.
