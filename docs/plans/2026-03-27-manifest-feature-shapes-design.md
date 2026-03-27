# Manifest Feature Shape Metadata Design

## Context

`load_partitioned_stores(...)` and `StoreBackedSamplingCoordinator.from_partition_dir(...)` now avoid eager partition payload deserialization and only load `part-*.pt` on first real data access. That fixed the main large-graph bottleneck, but one metadata hole remains: feature `shape(...)` still falls back to loading a partition payload because the shape lives only inside the serialized tensors.

That matters more than it looks. Empty feature fetch paths call `shape(...)` to allocate correctly typed empty outputs, so a zero-length request can still trigger partition payload loads. For a manifest-backed path, that is the wrong tradeoff.

## Goals

- Make partition feature `shape(...)` resolvable from manifest metadata without `torch.load(...)`
- Keep `PartitionedFeatureStore.shape(...)` lazy by default
- Keep `StoreBackedSamplingCoordinator.fetch_*_features(...)` manifest-only for empty requests
- Preserve existing partition payload format and fallback behavior

## Approach

Store per-partition feature shapes in `PartitionShard.metadata` alongside the existing edge-id metadata.

Use two metadata maps:

- `node_feature_shapes`: `node_type -> feature_name -> shape`
- `edge_feature_shapes`: `edge_type -> feature_name -> shape`

The partition writer already has the shard-local graph in memory, so it can emit these shapes cheaply while writing `manifest.json`. The partition metadata layer will normalize and serialize the new maps the same way it already does for edge-type-keyed metadata.

On the reader side, lazy feature adapters will consult manifest metadata for `shape(...)` first and only fall back to payload-backed `FeatureStore.shape(...)` if a key is genuinely missing from metadata.

## Non-Goals

- No partition payload schema rewrite
- No mmap-backed partition tensor format in this batch
- No boundary-edge-specific shape surface; existing edge feature shapes are sufficient for current empty-fetch paths

## Verification

- Regression proving `load_partitioned_stores(...).shape(...)` stays at zero payload loads
- Regression proving empty coordinator feature fetches stay manifest-only
- Manifest round-trip coverage for the new shape metadata
- Fresh focused and full `pytest` verification before merge
