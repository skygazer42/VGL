# Heterogeneous Random Link Split

## Goal

Extend `RandomLinkSplit` beyond homogeneous defaults so it can produce link-prediction datasets for typed relations in heterogeneous graphs.

## Design

`RandomLinkSplit` now supports:

- `edge_type`: target supervision relation to split
- `rev_edge_type`: optional reverse relation that should stay aligned with the target relation in message-passing graphs

When `edge_type` is set, records are created from that relation only, and each `LinkPredictionRecord` carries:

- `edge_type`
- optional `reverse_edge_type`

This keeps downstream negative samplers and neighbor samplers relation-aware without introducing new dataset types.

## Reverse Relation Handling

When `rev_edge_type` is provided, train/val/test message-passing graphs slice the reverse relation using reverse edge pairs of the selected target edges:

- train graph: target train edges + reverse(train)
- val graph: target train edges + reverse(train)
- test graph: target train(+val) edges + reverse(train(+val)) depending on `include_validation_edges_in_test`

## Seed-edge Exclusion

`LinkPredictionBatch` now also removes reverse supervision edges when a record includes:

- `exclude_seed_edges=True` (directly or via metadata)
- `reverse_edge_type`

This mirrors mainstream hetero link-prediction training setups where both directions of a supervised pair are removed from message passing.

## Validation

- `tests/data/test_random_link_split.py` covers hetero split with reverse relation alignment
- `tests/core/test_link_prediction_batch.py` covers reverse-edge exclusion for hetero records
- full test suite and lint pass
