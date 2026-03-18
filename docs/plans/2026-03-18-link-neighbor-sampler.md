# Link Neighbor Sampler

## Goal

Add a PyG/DGL-style link mini-batch sampling path for homogeneous graphs without redesigning the existing loader or task abstractions.

## Design

`LinkNeighborSampler` composes with the samplers already added earlier in the day:

- `UniformNegativeLinkSampler`
- `HardNegativeLinkSampler`
- `CandidateLinkSampler`

It works as:

```python
LinkNeighborSampler(
    num_neighbors=[15, 10],
    base_sampler=UniformNegativeLinkSampler(num_negatives=2),
)
```

The base sampler first expands one positive seed into supervision records. `LinkNeighborSampler` then:

1. collects the supervision endpoints used by those records
2. samples hop-wise neighbors around them
3. builds an induced local homogeneous subgraph
4. remaps `src_index` and `dst_index` into local node coordinates
5. preserves query/filter metadata for existing ranking metrics

## Batching

To make this usable with `Loader(batch_size > 1)`, `LinkPredictionBatch` now supports batching records from multiple local subgraphs by constructing a disjoint-union graph and offsetting supervision indices.

That keeps the current trainer/task contract intact while allowing multiple sampled edge queries per optimizer step.

## Scope

- homogeneous graphs only for the multi-graph batching path
- keeps the existing `Loader` contract
- no new evaluator or datamodule layer introduced

## Validation

- sampler unit tests cover local remapping and composition with negative sampling
- trainer tests cover one-epoch link prediction with neighbor sampling
- integration example now uses:
  `RandomLinkSplit -> LinkNeighborSampler(base_sampler=...) -> Trainer.fit/test`
