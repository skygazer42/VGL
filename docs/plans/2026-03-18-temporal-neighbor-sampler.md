# Temporal Neighbor Sampler

## Goal

Add a mainstream temporal mini-batch path similar to the sampled-history loaders used in PyG and DGL/GraphBolt, without introducing a new datamodule abstraction.

## Design

Two pieces are added together:

- `TemporalNeighborSampler`
- multi-graph `TemporalEventBatch`

`TemporalNeighborSampler` consumes `TemporalEventRecord` items and builds a local temporal subgraph per event. The sampler:

1. filters the source graph down to event history before the supervision timestamp
2. optionally restricts that history with `time_window`
3. optionally keeps only the most recent `max_events`
4. samples hop-wise neighbors around the event endpoints
5. remaps `src_index` and `dst_index` into local node coordinates
6. preserves original node ids as `n_id` and edge ids as `e_id`

The default behavior is strict history (`edge_time < event_time`) to avoid label leakage for temporal event prediction.

## Batching

`TemporalEventBatch` now supports records coming from multiple sampled temporal graphs by constructing a disjoint-union temporal graph and offsetting event endpoints into the batched node space.

That keeps the existing trainer/task contract intact:

- `batch.graph` is still the message-passing graph
- `batch.src_index` / `batch.dst_index` stay valid inside the batched graph
- `batch.history_graph(i)` still works by snapshotting the batched temporal graph at the event timestamp

## Scope

- homogeneous temporal graphs for the sampler
- no new trainer abstraction
- keeps existing full-graph temporal event prediction working

## Validation

- unit tests cover strict-history extraction and `max_events`
- batch tests cover multi-graph temporal batching
- loader, trainer, and integration tests now run sampled temporal event prediction end to end
