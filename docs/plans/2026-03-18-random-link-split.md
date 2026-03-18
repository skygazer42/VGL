# Random Link Split

## Goal

Add a mainstream link-prediction data split utility similar in spirit to PyG's `RandomLinkSplit`, but adapted to VGL's existing `LinkPredictionRecord -> DataLoader -> Sampler -> Trainer` pipeline.

## Design

`RandomLinkSplit` lives in `vgl.transforms` and returns:

```python
train_dataset, val_dataset, test_dataset = RandomLinkSplit(...)(graph)
```

Each dataset is a `ListDataset` of `LinkPredictionRecord` items, with optional split-time negatives.

By default, split datasets contain positives only, so you can still use sampler-time negatives:

- training-time negative sampling via `UniformNegativeLinkSampler` / `HardNegativeLinkSampler`
- evaluation-time candidate ranking via `CandidateLinkSampler`

When needed, split-time negatives can be enabled with:

- `neg_sampling_ratio`
- `add_negative_train_samples`

## Semantics

- `train_dataset` records point to a graph containing only train edges
- `val_dataset` records point to the train graph
- `test_dataset` records point to the train+val graph by default
- `is_undirected=True` keeps reverse-direction edge pairs in the same split group to avoid leakage
- `disjoint_train_ratio` can hold out a subset of train edges as supervision-only edges (removed from message passing)
- heterogeneous graphs are supported with `edge_type` and optional `rev_edge_type`

## Scope

Initial implementation targets the current homogeneous link-prediction workflow and avoids introducing a second task/evaluator abstraction.

## Validation

- unit tests cover split counts, validation/test message-passing graph semantics, undirected leakage prevention, hetero edge-type splits, disjoint train supervision, and split-time negative sampling
- integration test covers the full pipeline:
  `RandomLinkSplit -> UniformNegativeLinkSampler -> CandidateLinkSampler -> Trainer.fit/test`
