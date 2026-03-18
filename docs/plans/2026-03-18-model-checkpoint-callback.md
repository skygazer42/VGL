# Model Checkpoint Callback Plan

## Goal

Add a callback-based checkpointing strategy aligned with mainstream trainer stacks (for example PyTorch Lightning): periodic checkpointing with `save_top_k` and `save_last`.

## Scope

- Add `ModelCheckpoint` callback in `vgl.engine.callbacks`
- Support options:
  - `monitor`/`mode`
  - `save_top_k`
  - `save_last`
  - `save_on_exception`
  - `every_n_epochs`
  - configurable filename template
- Include callback state in training checkpoint resume flow
- Export callback from `vgl.engine`, `vgl.train`, `vgl.train.callbacks`, and `vgl`
- Add tests for:
  - configuration validation
  - top-k/last save behavior
  - callback resume equivalence
  - package exports

## Design

- Reuse `trainer.save_training_checkpoint(...)` so saved files keep full trainer state sections
- Save `last.ckpt` at configured epoch interval when enabled
- Save top-k checkpoints based on monitor metric and strict improvement over current worst kept checkpoint
- Keep callback state (`best_k_models`, best/kth paths and scores, active monitor/mode) for deterministic resume
- Automatically prune evicted top-k checkpoints from disk (except active `last.ckpt`)

## Verification

- `python -m pytest tests/train/test_callbacks.py tests/train/test_trainer_evaluation.py tests/test_package_exports.py tests/test_package_layout.py -q`
- `python -m pytest -q`
- `python -m ruff check vgl tests examples`
