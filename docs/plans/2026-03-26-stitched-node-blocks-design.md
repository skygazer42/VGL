# Distributed Stitched Node Block Output Design

## Context

VGL now supports stitched homogeneous node sampling through `LocalSamplingCoordinator`, and it also supports optional DGL-style `blocks` output for local homogeneous `NodeNeighborSampler(..., output_blocks=True)`. Those two capabilities stop just short of each other: as soon as node sampling switches to the stitched coordinator path, `output_blocks=True` still fails with an explicit unsupported error.

That leaves a real large-graph training gap versus DGL-class workflows. A shard-local node batch can now stitch remote frontier structure into one sampled subgraph, but it still cannot emit layer-wise message-flow blocks for that stitched neighborhood.

## Scope Choice

Three slices were considered:

1. Add stitched block output for node, link, and temporal samplers together.
2. Add stitched block output only for homogeneous node sampling first.
3. Skip stitched homogeneous and go straight to heterogeneous block output.

Option 2 is the right next slice. The stitched homogeneous node path already exists, the `Block` abstraction is homogeneous-relation friendly today, and node mini-batches already know how to batch blocks. Doing node/link/temporal together would widen the change too much, while heterogeneous block output still needs a larger multi-relation block representation.

## Recommended API

Keep the public API additive and unchanged:

- `NodeNeighborSampler(..., output_blocks=False)`
- `NodeNeighborSampler(..., output_blocks=True)`
- `SampleRecord.blocks: list[Block] | None`
- `NodeBatch.blocks: list[Block] | None`

New semantics:

- local homogeneous sampling keeps the existing block behavior
- stitched homogeneous sampling through `Loader(..., feature_store=coordinator)` or `PlanExecutor.execute(..., feature_store=coordinator)` now also materializes `blocks`
- the sampled `NodeBatch.graph` and `NodeBatch.seed_index` contract stays unchanged
- blocks remain ordered outer-to-inner
- heterogeneous stitched sampling still fails clearly for `output_blocks=True`

## Construction Strategy

Blocks should still be built from the sampled graph, not from raw coordinator incident-edge queries. The stitched path already constructs one stitched homogeneous sampled graph with relabeled local `edge_index` and global `n_id` / `e_id`. The missing piece is retaining cumulative hop frontiers in global-id space so materialization can turn that stitched sampled graph into blocks after any fetched feature overlays have been applied.

The stitched branch in `PlanExecutor._expand_neighbors(...)` should therefore:

1. Expand seeds in global-id space exactly as it does today.
2. When `output_blocks=True`, also retain one cumulative hop tensor per expansion step, still in global-id space.
3. Store those hop tensors in executor state beside the sampled stitched graph.
4. Let materialization rebuild blocks from the stitched sampled graph by mapping each hop's global ids back into sampled-graph local positions through the graph's `n_id` order.

This mirrors the local homogeneous block path, but uses stitched graph `n_id` values as the public/global identity space.

## Feature Alignment

The stitched graph already materializes aligned tensor features directly from the coordinator, and later plan-backed fetch stages can still overlay additional tensors through `n_id` / `e_id`. Blocks must be created after that overlay step, not in the executor, so block graphs inherit the final sampled-graph tensor state just like the local homogeneous path.

## Validation And Deferred Scope

This batch should remove exactly one unsupported case:

- stitched homogeneous node sampling with `output_blocks=True`

This batch should continue to fail clearly for:

- stitched heterogeneous node sampling with `output_blocks=True`
- stitched homogeneous link sampling with `output_blocks=True`
- stitched temporal block output

This batch should not change:

- the `Loader` return type
- `NodeBatch.seed_index` semantics
- link or temporal block materialization
- heterogeneous block abstractions

## Testing Strategy

Add focused regressions for:

- stitched homogeneous node loading with `output_blocks=True`
- preservation of remote frontier global `n_id` / `e_id` inside stitched blocks
- fixed block count across the configured hop list on stitched paths
- block feature overlay compatibility when plan-backed node or edge fetch stages run through the coordinator
- the existing unsupported heterogeneous stitched case staying explicit

Then rerun the focused node/block/coordinator suite and the full repository suite before merge.
