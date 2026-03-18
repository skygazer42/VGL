# Hard Negative Link Sampler Plan

## Goal

Add a dedicated hard-negative sampler for link prediction that prioritizes record-provided negative destinations and backfills the remaining quota with uniform negatives.

## Reference Direction

- GraphStorm treats hard negatives as first-class link-prediction training inputs.
- DGL GraphBolt exposes hard-negative-aware minibatch construction alongside seed-edge exclusion.
- PyG loaders keep supervision edges, negative sampling, and message-passing structure separated instead of hiding this logic inside the task loss.

## Scope

- Add `HardNegativeLinkSampler` in `vgl.dataloading.sampler`
- Extend `LinkPredictionRecord` with optional hard-negative destinations
- Re-export the sampler through `vgl.dataloading`, `vgl.data`, and `vgl`
- Add tests for:
  - hard-negative priority over uniform backfill
  - optional seed-edge exclusion
  - invalid hard-negative indices
  - trainer integration
  - package exports

## Design

- Keep the seed dataset made of positive `LinkPredictionRecord` items
- Read hard-negative destinations from `record.hard_negative_dst`
- Sample up to `num_hard_negatives` valid hard negatives first
- Fill any remaining `num_negatives` budget with the same uniform-negative path used by `UniformNegativeLinkSampler`
- Reuse the existing seed-edge exclusion path for leakage prevention

## Verification

- `python -m pytest tests/data/test_link_prediction_loader.py tests/train/test_link_prediction_trainer.py tests/test_package_exports.py tests/test_package_layout.py -q`
- `python -m pytest -q`
- `python -m ruff check vgl tests examples`
