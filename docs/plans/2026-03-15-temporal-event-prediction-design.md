# Temporal Event Prediction Phase 3 Design

**Date:** 2026-03-15
**Status:** Approved for planning

## Goal

Extend the current framework with a stable temporal event prediction training path for single-event-type temporal graphs, using explicit candidate-event samples as the public data contract.

## Scope Decisions

- Focus on `temporal event prediction`, not on general temporal operator breadth
- Prioritize a `single event type` path before heterogenous temporal training
- Use `explicit candidate events` as the dataset contract
- Keep one canonical `Graph` abstraction and extend the existing view-based temporal model
- Preserve the current node classification and graph classification paths without regressions

## Chosen Direction

Three directions were considered:

1. Reuse existing `GraphBatch` and place temporal supervision inside `metadata`
2. Add a dedicated temporal sample and batch contract while keeping `Trainer` unchanged
3. Reframe event prediction as graph classification over synthetic per-event graphs

The chosen direction is:

> Add dedicated `TemporalEventRecord`, `TemporalEventBatch`, and `TemporalEventPredictionTask` contracts while preserving the existing `Graph` abstraction and generic `Trainer`.

This is the only direction that keeps the temporal training surface explicit and stable. Hiding temporal semantics inside generic metadata would make the API harder to evolve, and pretending event prediction is graph classification would distort both concepts.

## Architecture

Phase 3 should extend the current pipeline as follows:

`Dataset/Sampler -> TemporalEventRecord -> Loader -> TemporalEventBatch -> Temporal Encoder/Model -> TemporalEventPredictionTask`

The graph core remains unchanged. Temporal semantics continue to rely on `graph.schema.time_attr` plus `graph.snapshot()` and `graph.window()` rather than introducing a separate top-level graph family.

## Core Additions

### TemporalEventRecord

`TemporalEventRecord` should be the smallest structured unit that flows into temporal event prediction collation.

Each record should carry at least:

- `graph`
- `src_index`
- `dst_index`
- `timestamp`
- `label`
- optional `event_features`
- optional `metadata`
- optional `sample_id`

This keeps temporal supervision explicit and avoids burying core fields inside loose metadata dictionaries.

### TemporalEventBatch

`TemporalEventBatch` should be the single object passed to models and tasks for temporal event prediction.

The batch should expose at least:

- `graph`
- `src_index`
- `dst_index`
- `timestamp`
- `labels`
- optional `event_features`
- optional `metadata`

The first implementation should support the common case where all samples in a batch come from one source graph. The batch may provide helper access for per-sample history views, but it should not attempt advanced multi-graph temporal collation in this phase.

### TemporalEventPredictionTask

Phase 3 should introduce `TemporalEventPredictionTask` as the supervision contract for temporal classification-style event prediction.

The task should:

- accept explicit labeled event batches
- compute classification loss
- reject unsupported loss modes early

The stable path is a classification contract over explicit candidate events. Ranking, automatic negative sampling, and regression-style temporal prediction are out of scope for this phase.

## Data Flow

Phase 3 should use explicit candidate-event samples.

Each dataset item or sampler output should describe a candidate event:

- source node index
- destination node index
- event time
- label
- optional event features

Loader collation should preserve these fields in `TemporalEventBatch` and ensure that event context is computed from historical graph state only.

## Batch Semantics

The first batch design should favor API stability over aggressive optimization.

For a batch of explicit candidate events:

- retain the shared temporal graph reference
- store vectorized `src_index`, `dst_index`, `timestamp`, and `labels`
- optionally store `event_features` and `metadata`

Context construction must remain causal. When a model or helper builds the history view for an event at time `t`, it must only see edges whose timestamp satisfies `<= t`.

The phase should not yet optimize repeated history extraction across many event times. Repeated `snapshot()` calls are acceptable for the first stable version.

## Public API Shape

The public training API should remain structurally similar to existing tasks:

```python
task = TemporalEventPredictionTask(
    target="label",
    loss="cross_entropy",
)
trainer = Trainer(
    model=model,
    task=task,
    optimizer=torch.optim.Adam,
    lr=1e-3,
    max_epochs=5,
)
trainer.fit(loader)
```

Models should consume one normalized batch object:

```python
logits = model(batch)
loss = task.loss(batch, logits, stage="train")
```

The trainer should not gain temporal-specific calling conventions in this phase.

## Error Handling

Temporal event prediction should fail early when:

- the source graph is not temporal and has no `schema.time_attr`
- any record omits `src_index`, `dst_index`, `timestamp`, or `label`
- `src_index` or `dst_index` falls outside the node range
- batch field lengths disagree
- `event_features` does not align with batch size
- a collated batch mixes multiple source graphs in the single-graph path
- an unsupported loss mode is requested

Error messages should be explicit enough to distinguish data contract failures from model failures.

## Testing Strategy

Phase 3 tests should cover four layers:

### Contract tests

- `TemporalEventRecord` validation
- `TemporalEventBatch` field alignment and shape guarantees
- loader collation for explicit candidate-event samples

### Task and trainer tests

- `TemporalEventPredictionTask` computes loss on temporal event batches
- `Trainer.fit(loader)` runs a temporal event prediction epoch end to end
- invalid loss modes and invalid graph kinds fail early

### Temporal semantics tests

- history views only include edges with timestamp `<= t`
- future events do not leak into the event context

### Example and documentation tests

- `examples/temporal/event_prediction.py` becomes a real event prediction example
- README and quickstart remain accurate and runnable

## Phase 3 Deliverables

Phase 3 should ship:

- `TemporalEventRecord`
- `TemporalEventBatch`
- `TemporalEventPredictionTask`
- loader support for explicit candidate-event samples
- one real temporal event prediction example
- matching unit and integration tests
- documentation updates

## Explicit Non-Goals for Phase 3

Do not include:

- heterogeneous temporal event training
- automatic negative sampling
- ranking losses and ranking metrics
- full-event tuple generation `(src, rel, dst, time)`
- multi-graph temporal batch optimization
- a broad temporal operator zoo

These belong to later phases and should not distort the first stable temporal training contract.

## Repository Touchpoints

Phase 3 will mostly affect:

- `src/gnn/core/`
- `src/gnn/data/`
- `src/gnn/train/`
- `examples/temporal/`
- `docs/`
- `tests/data/`
- `tests/train/`
- `tests/integration/`

## Stability Constraints

Phase 3 should follow three rules:

1. Keep one canonical `Graph` abstraction
2. Keep temporal supervision fields explicit rather than hiding them in generic metadata
3. Enforce causal history views so future events never appear in model context

These constraints keep the API coherent as later phases add heterogenous temporal support or more advanced event objectives.

## Acceptance Criteria

Phase 3 is complete when:

1. explicit candidate-event samples train through `Trainer` end to end
2. temporal event batches expose stable public fields
3. history construction does not leak future edges
4. existing node classification and graph classification tests still pass
5. the temporal example demonstrates real event prediction instead of placeholder node classification

## Next Step

The next step is to create a detailed implementation plan with exact files, tests, commands, and commit checkpoints.
