# Link Prediction Ranking Metrics Plan

## Goal

Add mainstream link-prediction ranking metrics so VGL can evaluate sampled candidate sets with the same baseline metrics commonly used in PyG, DGL, and GraphStorm workflows.

## Reference Direction

- GraphStorm highlights ranking metrics such as `MRR` and `Hits@K` for link prediction.
- DGL link-prediction tutorials and examples commonly report reciprocal-rank and hit-rate style metrics.
- PyG evaluation flows also treat ranking metrics as the standard complement to BCE-style training losses.

## Scope

- Add `MRR` and `HitsAtK` metric classes in `vgl.metrics`
- Extend `build_metric()` with `mrr` and `hits@K`
- Extend link-prediction records / batches with query grouping metadata
- Make negative samplers emit query grouping automatically
- Update trainer metric dispatch so metrics can inspect the current batch
- Re-export the new metrics through `vgl.metrics`, `vgl.train.metrics`, `vgl.train`, and `vgl`

## Design

- Keep `LinkPredictionTask` loss unchanged
- Treat each positive seed edge and its sampled negatives as one ranking query
- Store a `query_id` on sampled link records and convert it into `batch.query_index`
- Let ranking metrics use `batch.query_index` to compute per-query ranks
- Keep behavior unchanged for tasks and batches that do not use ranking metrics

## Verification

- `python -m pytest tests/train/test_metrics.py tests/data/test_link_prediction_loader.py tests/train/test_link_prediction_trainer.py tests/test_package_exports.py tests/test_package_layout.py -q`
- `python -m pytest -q`
- `python -m ruff check vgl tests examples`
