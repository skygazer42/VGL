# Link Neighbor Sampler Block Output Design

## Context

VGL now supports relation-local `Block`, `to_block(...)`, and local homogeneous `NodeNeighborSampler(..., output_blocks=True)`, but link prediction mini-batches still expose only a sampled subgraph plus endpoint indices. For DGL-style edge prediction workflows, this leaves a practical gap: link supervision can sample a local subgraph, but it still cannot expose layer-wise message-flow blocks for the sampled message-passing graph.

That gap matters because many link prediction models want two separate views of one batch:

- the supervision pairs through `src_index` / `dst_index`
- the message-passing frontier through stacked blocks

Keeping only the sampled subgraph forces downstream code to rebuild blocks manually and re-encode edge-exclusion semantics outside the loader.

## Scope Choice

Three plausible slices were considered:

1. Add `blocks` to all link sampling modes at once, including heterogeneous and stitched/distributed graphs.
2. Add `blocks` only for local homogeneous `LinkNeighborSampler`, keep the existing `LinkPredictionBatch` contract, and carry blocks as an optional extra field.
3. Replace `LinkPredictionBatch.graph` with blocks entirely.

Option 2 is the correct slice. Replacing `batch.graph` would break existing task/model code. Heterogeneous and stitched block output need a larger cross-relation or cross-partition message-flow abstraction. Local homogeneous link sampling is the highest-value minimal extension and reuses the block substrate already proven on node sampling.

## Recommended API

Extend:

- `LinkNeighborSampler(..., output_blocks=False)`
- `LinkPredictionRecord.blocks: list[Block] | None`
- `LinkPredictionBatch.blocks: list[Block] | None`

Semantics:

- when `output_blocks=False`, link sampling behavior stays unchanged
- when `output_blocks=True` on local homogeneous link sampling, sampled records carry one shared block list and `LinkPredictionBatch` carries one batched block list
- block order is outer-to-inner
- `batch.graph`, `batch.src_index`, and `batch.dst_index` remain the public supervision surface

## Block Construction Strategy

The seed set for link sampling is the union of all supervised endpoints in the sampled record set. For a fanout list of length `L`, the sampler should record cumulative homogeneous frontier states:

- hop 0: the supervised endpoint union
- hop 1: cumulative visited nodes after the first expansion
- ...
- hop L: all nodes in the sampled subgraph

Blocks should then be built from the sampled subgraph in reverse cumulative order, exactly like node sampling:

- block 0 uses hop `L - 1`
- ...
- block `L - 1` uses hop 0

Blocks must be built from the sampled subgraph, not the full source graph, so fanout constraints remain intact.

## Message-Passing Edge Exclusion

Link prediction already distinguishes between the sampled supervision graph and the message-passing graph through `exclude_seed_edges`. When `output_blocks=True`, blocks must follow message-passing semantics rather than raw sampled-subgraph semantics.

That means:

- sampled `record.graph` can stay unchanged, preserving current behavior
- `record.blocks` and `batch.blocks` should be built from the sampled graph after local seed-edge exclusion has been applied
- positive supervision edges excluded from `batch.graph` should also be absent from the block edge sets

This keeps `blocks` aligned with the real message-passing path used during training.

## Validation And Deferred Scope

This batch should fail clearly when `output_blocks=True` is requested for unsupported cases:

- heterogeneous link sampling
- stitched/distributed homogeneous link sampling

This batch should not change:

- `LinkPredictionTask` semantics
- heterogeneous or mixed-edge-type block abstractions
- distributed block materialization
- loader return types

## Testing Strategy

Add focused tests for:

- homogeneous `LinkNeighborSampler(..., output_blocks=True)` record output
- outer-to-inner block ordering and preserved `n_id` / `e_id`
- sampled-edge-only behavior under finite fanout
- `exclude_seed_edges` removing supervision edges from blocks
- loader materialization exposing `LinkPredictionBatch.blocks`
- `LinkPredictionBatch.from_records(...)` batching per-layer blocks
- `LinkPredictionBatch.to(...)` and `LinkPredictionBatch.pin_memory()` carrying blocks
- clear errors for unsupported heterogeneous block output

Then rerun the full suite before merge.
