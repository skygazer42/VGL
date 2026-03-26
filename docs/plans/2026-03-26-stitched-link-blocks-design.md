# Distributed Stitched Link Block Output Design

## Context

VGL now supports stitched homogeneous link sampling through `LocalSamplingCoordinator`, and it also supports optional DGL-style `blocks` output for local homogeneous `LinkNeighborSampler(..., output_blocks=True)`. Those two capabilities still stop short of each other: once link sampling switches to the stitched coordinator path, `output_blocks=True` fails with an explicit unsupported error.

That leaves a real large-graph training gap versus DGL-class workflows. A shard-local link batch can already stitch remote frontier structure into one sampled subgraph, but it still cannot emit layer-wise message-flow blocks for that stitched message-passing view.

## Scope Choice

Three slices were considered:

1. Add stitched block output for homogeneous, heterogeneous, and temporal link samplers together.
2. Add stitched block output only for homogeneous link sampling first.
3. Skip homogeneous stitched support and jump straight to heterogeneous stitched link blocks.

Option 2 is the right next slice. The stitched homogeneous link path already exists, the local homogeneous link block contract already exists, and the current `Block` abstraction is still homogeneous-relation friendly. Pulling hetero or temporal link blocks into the same batch would widen the change and force decisions about richer multi-relation block contracts that are not needed for this gap.

## Recommended API

Keep the public API additive and unchanged:

- `LinkNeighborSampler(..., output_blocks=False)`
- `LinkNeighborSampler(..., output_blocks=True)`
- `LinkPredictionRecord.blocks: list[Block] | None`
- `LinkPredictionBatch.blocks: list[Block] | None`

New semantics:

- local homogeneous link sampling keeps the existing block behavior
- stitched homogeneous link sampling through `Loader(..., feature_store=coordinator)` or `PlanExecutor.execute(..., feature_store=coordinator)` now also materializes `blocks`
- the sampled `LinkPredictionBatch.graph`, `src_index`, `dst_index`, `labels`, and seed-edge exclusion semantics stay unchanged
- blocks remain ordered outer-to-inner
- heterogeneous stitched link sampling still fails clearly for `output_blocks=True`

## Construction Strategy

Blocks should still be built from the sampled message-passing graph, not from raw coordinator incident-edge queries. The stitched homogeneous link path already constructs one stitched sampled graph with relabeled local `edge_index` plus global `n_id` / `e_id`, and materialization already knows how to derive the message-passing-only graph by removing excluded supervision edges before building blocks.

The missing piece is retaining cumulative link frontier states in global-id space on the stitched executor branch. The stitched homogeneous link path should therefore:

1. Expand the seed endpoints in global-id space exactly as it does today.
2. When `output_blocks=True`, also retain one cumulative hop tensor per expansion step, still in global-id space.
3. Store those hop tensors in executor state beside the stitched sampled graph, along with the stitched graph `n_id` ordering used to map global ids back to sampled local positions.
4. Let materialization rebuild blocks from the stitched message-passing graph after feature overlays and seed-edge exclusion have been applied.

This mirrors the local homogeneous link block path, but uses stitched graph `n_id` values as the identity space for hop reconstruction.

## Feature Alignment And Seed-Edge Exclusion

The stitched graph already materializes aligned features from the coordinator, and later plan-backed fetch stages can still overlay node or edge tensors through sampled `n_id` / `e_id`. Blocks must continue to be created during materialization, not in the executor, so block graphs inherit the final sampled-graph tensor state.

Link blocks also have one extra constraint versus node blocks: they are built from the message-passing graph after optional seed-edge exclusion. The stitched path should therefore reuse the existing `_link_message_passing_graph(...)` branch and only change how hop frontiers are mapped onto that graph.

## Validation And Deferred Scope

This batch should remove exactly one unsupported case:

- stitched homogeneous link sampling with `output_blocks=True`

This batch should continue to fail clearly for:

- stitched heterogeneous link sampling with `output_blocks=True`
- stitched node or temporal behavior outside the already implemented scope
- any heterogeneous block materialization path

This batch should not change:

- the `Loader` return type
- `LinkPredictionBatch` supervision fields
- local homogeneous block semantics
- heterogeneous or temporal block abstractions

## Testing Strategy

Add focused regressions for:

- stitched homogeneous link loading with `output_blocks=True`
- preservation of remote frontier global `n_id` / `e_id` inside stitched blocks
- fixed block count across the configured hop list on stitched paths even when a later hop adds no new nodes
- seed-edge exclusion still removing supervision edges from stitched block message passing
- coordinator-backed node and edge feature overlays remaining visible on stitched block graphs after materialization
- the existing unsupported heterogeneous stitched case staying explicit

Then rerun focused link/block/coordinator suites and the full repository suite before merge.
