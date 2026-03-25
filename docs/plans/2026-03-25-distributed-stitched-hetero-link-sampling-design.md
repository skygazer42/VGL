# Distributed Stitched Hetero Link Sampling Design

## Problem

VGL can now stitch shard-local homogeneous node, link, and temporal sampling across partition boundaries, and it can also stitch shard-local non-temporal heterogeneous node sampling through `LocalSamplingCoordinator`. The remaining non-temporal distributed sampling gap is heterogeneous `LinkNeighborSampler`: a shard-local hetero supervision edge can fetch routed features, but its sampled message-passing graph still expands only on the local shard graph and therefore misses remote typed frontier structure once any relation crosses a partition boundary.

That gap now sits squarely in the executor layer rather than in storage or partition metadata. The coordinator already exposes the typed routing and incident-edge queries needed to expand heterogeneous frontiers in global-id space, and the executor already knows how to build stitched hetero graphs for node sampling plus stitched homogeneous graphs for link sampling. The missing piece is combining those two capabilities for link records.

## Goals

- Add stitched non-temporal heterogeneous `LinkNeighborSampler` support for shard-local graphs when the loader uses a coordinator-backed feature source
- Keep the public `LinkNeighborSampler`, `Loader`, `LinkPredictionBatch`, and coordinator APIs stable for existing callers
- Preserve the current local-only behavior for full graphs, homogeneous graphs, temporal graphs, and non-coordinator feature sources
- Keep sampled typed `n_id` / `e_id` globally aligned so routed feature materialization continues to work without special cases
- Preserve current record batching semantics, including multiple records from one source graph and mixed hetero edge types from a base sampler when their endpoints can share one stitched graph

## Recommended Design

Keep the public link-sampler plan unchanged and extend `PlanExecutor._sample_link_neighbors(...)` with one stitched heterogeneous branch that activates after the existing stitched homogeneous branch and before the local fallback.

That branch should activate only when:

- all seed records come from the same non-temporal heterogeneous shard-local graph
- the executor feature source is a coordinator-backed routed source for the exact shard that produced that graph
- every node store on the shard graph carries global `n_id`
- the source graph matches exactly one coordinator partition under the same shard guard used by stitched hetero node sampling

When the guard matches, the executor should:

1. Collect all record endpoints into typed local seed sets keyed by node type, using each record's resolved `edge_type` to determine the source and destination node types.
2. Convert those typed local seeds into typed global node ids through the shard graph's per-type `n_id` tensors.
3. Expand the union frontier hop-by-hop in coordinator global-id space with the same per-hop fanout semantics already used by stitched hetero node sampling.
4. Collect typed incident edges whose endpoints both stay inside the visited node set, deduplicate them by typed global `e_id`, and build one stitched hetero graph with relabeled typed `edge_index`, typed global `n_id`, typed global `e_id`, and aligned node/edge tensors fetched through the coordinator.
5. Rebuild each `LinkPredictionRecord` so its `src_index` and `dst_index` point into the stitched graph's per-type node spaces while preserving labels, metadata, sample ids, query ids, edge-type metadata, and current link-task flags.
6. Store sampled typed node and edge ids in executor state exactly the same way as stitched hetero node sampling so later feature-fetch stages continue to align against global ids.

A small refactor is justified here: factor the stitched hetero frontier expansion into a global-id helper that accepts arbitrary typed seed sets, then let hetero node sampling call that helper as the one-node-type special case. That keeps node and link stitched expansion semantics identical and avoids maintaining two parallel hetero frontier walkers.

## Non-Goals

- No stitched heterogeneous temporal sampling in this batch
- No distributed negative-sampler redesign or DistDGL-style remote execution model
- No redesign of `LinkPredictionBatch`, `LinkPredictionTask`, or record schemas
- No partition-writer format changes
- No mutation of core `Graph` objects with partition metadata

## Verification

- Link-sampler regression proving a shard-local heterogeneous supervision edge can stitch remote typed frontier nodes and relation edges through the coordinator
- Public sampled link-training regression proving `Loader(..., sampler=LinkNeighborSampler(...), feature_store=coordinator)` can train against the stitched hetero graph path
- Focused sampler/materialization regressions for node, link, temporal, and local partition training surfaces
- Fresh full repository regression before merge
