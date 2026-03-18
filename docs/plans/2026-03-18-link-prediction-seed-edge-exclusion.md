# Link Prediction Seed Edge Exclusion Plan

## Goal

Prevent link-prediction training batches from leaking supervision edges into the message-passing graph, following the same high-level pattern used by mainstream GNN training stacks.

## Reference Direction

- PyG link loaders separate supervision edges from sampled message-passing neighborhoods.
- DGL GraphBolt exposes explicit seed-edge exclusion in minibatch link prediction flows.
- GraphStorm treats leakage prevention around target edges and negative sampling as first-class training-pipeline concerns.

## Scope

- Extend `UniformNegativeLinkSampler` with an `exclude_seed_edges` option
- Mark positive supervision records so batch collation can identify which edges to hide from message passing
- Update `LinkPredictionBatch.from_records()` to return a graph view with those seed edges removed
- Add tests for:
  - batch-level seed-edge exclusion
  - loader integration
  - trainer integration

## Design

- Keep the user-facing dataset made of positive `LinkPredictionRecord` seeds
- Preserve candidate edges and labels in the supervision tensors
- Build the message-passing graph as a view over the original graph with matching positive seed edges masked out
- Leave behavior unchanged unless `exclude_seed_edges=True`

## Verification

- `python -m pytest tests/core/test_link_prediction_batch.py tests/data/test_link_prediction_loader.py tests/train/test_link_prediction_trainer.py -q`
- `python -m pytest -q`
- `python -m ruff check vgl tests examples`
