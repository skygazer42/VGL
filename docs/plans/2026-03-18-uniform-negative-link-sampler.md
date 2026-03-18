# Uniform Negative Link Sampler Plan

## Goal

Add a dataloading sampler that expands positive link records into mixed positive/negative link-prediction minibatches, matching the negative-sampling-first training flow used by mainstream GNN packages.

## Scope

- Add `UniformNegativeLinkSampler` in `vgl.dataloading.sampler`
- Extend `Loader` so a sampler can expand one seed item into multiple collated records
- Re-export the sampler through `vgl.dataloading`, `vgl.data`, and `vgl`
- Add tests for:
  - loader collation with positive seed records plus sampled negatives
  - rejection of non-positive link seeds
  - trainer integration with `LinkPredictionTask`
  - package exports

## Design

- Treat dataset items as positive `LinkPredictionRecord` seeds
- Keep `batch_size` defined in terms of seed records, not expanded negative records
- For each positive seed, preserve the original edge and append `num_negatives` uniformly sampled destination corruptions with label `0`
- Optionally exclude destinations that already appear as outgoing positive edges for the same source node
- Raise a clear error if no valid negative destination exists

## Verification

- `python -m pytest tests/data/test_link_prediction_loader.py tests/train/test_link_prediction_trainer.py tests/test_package_exports.py tests/test_package_layout.py -q`
- `python -m pytest -q`
- `python -m ruff check vgl tests examples`
