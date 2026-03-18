# Filtered Link Prediction Ranking Metrics Plan

## Goal

Add filtered ranking metrics for link prediction so VGL can report the same evaluation family commonly used in mainstream GNN systems when candidate lists contain other known true edges.

## Reference Direction

- GraphStorm and DGL workflows commonly distinguish raw and filtered ranking evaluation for link prediction / KGE-style tasks.
- PyG evaluation code also treats candidate filtering as an evaluation concern rather than a training-loss concern.

## Scope

- Add `FilteredMRR` and `FilteredHitsAtK`
- Extend `LinkPredictionRecord` with a `filter_ranking` flag
- Extend `LinkPredictionBatch` with a boolean `filter_mask`
- Re-export the metrics through `vgl.metrics`, `vgl.train.metrics`, `vgl.train`, and `vgl`
- Add tests for:
  - filtered metric computation
  - batch filter-mask propagation
  - trainer integration
  - package exports

## Design

- Keep raw `MRR` / `HitsAtK` unchanged
- Treat `filter_mask=True` candidates as ignored when computing filtered ranks
- Require exactly one positive item per query after filtering
- Keep metric grouping driven by existing `batch.query_index`

## Verification

- `python -m pytest tests/core/test_link_prediction_batch.py tests/train/test_metrics.py tests/train/test_link_prediction_trainer.py tests/test_package_exports.py tests/test_package_layout.py -q`
- `python -m pytest -q`
- `python -m ruff check vgl tests examples`
