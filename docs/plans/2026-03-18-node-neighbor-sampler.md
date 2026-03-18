# Node Neighbor Sampler

## Goal

Add a mainstream mini-batch node classification path similar to PyG `NeighborLoader` and DGL neighbor sampling, while preserving the current full-graph training API.

## Design

Two pieces are added together:

- `NodeNeighborSampler`
- `NodeBatch`

`NodeNeighborSampler` consumes seed-node items in the existing loader style:

```python
(graph, {"seed": node_id, "sample_id": ...})
```

For each seed it samples hop-wise neighbors, builds a local homogeneous subgraph, stores global node ids as `n_id`, and records the remapped local supervision node as `subgraph_seed`.

When a loader batches those sampled `SampleRecord` objects without `label_source`, it now builds a `NodeBatch`:

- `graph`: disjoint-union graph over all sampled subgraphs in the mini-batch
- `seed_index`: local indices of the supervised nodes inside that batched graph

## Task Integration

`NodeClassificationTask` now supports both:

- full-graph training using stage masks
- sampled mini-batches using `NodeBatch.seed_index`

That keeps existing examples working while enabling a loader-driven mini-batch path.

## Scope

- homogeneous node mini-batches
- no new datamodule abstraction
- no changes required to the trainer loop contract

## Validation

- unit tests cover sampled node subgraphs and `NodeBatch`
- task contract tests cover `NodeClassificationTask` on `NodeBatch`
- end-to-end integration now runs node classification with neighbor-sampled loaders
