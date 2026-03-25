# Distributed Sampled Feature Alignment Design

## Problem

VGL's local distributed foundation can already partition graphs, reload shard-local `Graph` objects, route global node and edge ids through `LocalSamplingCoordinator`, and forward that coordinator into plan-backed feature fetch stages. The remaining gap is id alignment once sampling starts from a shard graph. `NodeNeighborSampler` expands neighborhoods against shard-local node ids, then fetch stages still read those staged ids verbatim. A coordinator-backed source interprets them as global ids, so sampled feature materialization silently points at the wrong rows whenever a shard owns non-zero global ids.

Link and temporal sampling are closer because their sampled subgraphs already carry `n_id` / `e_id`, but the runtime should make the distinction explicit rather than relying on each stage to know whether its current ids are local or global.

## Goals

- Keep the current `NodeNeighborSampler`, `LinkNeighborSampler`, `TemporalNeighborSampler`, `Loader`, and `PlanExecutor` public APIs unchanged
- Make coordinator-backed feature fetch work correctly for sampled shard graphs without manual id conversion wrappers
- Preserve existing direct `FeatureStore.fetch(...)` behavior for storage-backed and in-process graph workflows
- Prove the fix through sampled node, link, and temporal regressions plus one public loader integration path

## Recommended Design

The minimal fix lives inside the plan runtime.

1. Neighbor expansion should keep its current local-id state for subgraph materialization, but also record globalized node and edge ids when the source graph exposes `n_id` and `e_id`.
2. Feature fetch stages should distinguish between two source families:
   - direct store-style sources exposing `.fetch(...)`, which continue to consume the staged ids as-is
   - routed sources exposing `.fetch_node_features(...)` / `.fetch_edge_features(...)`, which should prefer the staged global-id views when present
3. Sampled link and temporal stages already record `n_id` / `e_id` from the sampled graph, so they can continue using the same state keys; the executor only needs to avoid re-translating ids that are already global.

This keeps local subgraph construction unchanged, keeps feature routing logic centralized in `PlanExecutor`, and removes the need for ad-hoc wrappers like the current integration-only routed loader pattern.

## Non-Goals

- No cross-partition stitched sampling
- No new distributed sampler classes or dataset record types
- No RPC, remote executors, or DistDGL-style global graph traversal
- No change to graph partition payload formats

## Verification

- Node-sampling regression proving `Loader(..., feature_store=coordinator)` overlays the correct global node and edge features onto sampled shard subgraphs
- Link-sampling regression proving sampled records fetched through the coordinator keep `n_id` / `e_id` aligned without double translation
- Temporal-sampling regression proving strict-history sampled shard subgraphs can fetch routed features through the same runtime path
- Integration regression demonstrating partition-local training can use the public loader path without a custom routed wrapper
- Fresh full repository regression before merge
