# Gradient Accumulation Scheduler Plan

## Goal

Add a callback-based training strategy that schedules `Trainer.accumulate_grad_batches` across epochs, aligned with mainstream trainer behavior (for example, dynamic accumulation scheduling in PyTorch Lightning).

## Scope

- Add `GradientAccumulationScheduler` callback in `vgl.engine.callbacks`
- Support callback export from `vgl.engine`, `vgl.train`, `vgl.train.callbacks`, and `vgl`
- Add tests for:
  - callback configuration validation
  - epoch-wise accumulation behavior and original-value restore
  - callback checkpoint resume behavior
  - package exports

## Design

- Accept `scheduling` as epoch-to-accumulation mapping (`{start_epoch: accumulate_grad_batches}`)
- Resolve the active accumulation value for each epoch by latest matching epoch key
- Apply schedule on `on_fit_start` and `on_epoch_end` for next epoch
- Preserve and restore the trainer's original `accumulate_grad_batches` on fit end

## Verification

- `python -m pytest tests/train/test_callbacks.py tests/train/test_trainer_evaluation.py tests/test_package_exports.py tests/test_package_layout.py -q`
- `python -m pytest -q`
- `python -m ruff check vgl tests examples`
