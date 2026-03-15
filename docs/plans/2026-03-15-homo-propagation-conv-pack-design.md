# Homogeneous Propagation Convolution Pack Design

**Date:** 2026-03-15
**Status:** Approved for planning

## Goal

Expand `vgl` with a new batch of homogeneous propagation-oriented operators by adding `AGNNConv`, `LightGCNConv`, and `FAGCNConv` while preserving the current stable graph and training entrypoints.

## Scope Decisions

- Focus on homogeneous graph operators only
- Add one coherent propagation batch:
  - `AGNNConv`
  - `LightGCNConv`
  - `FAGCNConv`
- Keep the public API aligned with current `vgl` conv usage patterns
- Continue supporting both `conv(graph)` and `conv(x, edge_index)`
- Treat the three operators as equal-width propagation modules
- Reuse the existing training path and `conv_zoo` example surface

## Chosen Direction

Three directions were considered:

1. Add more projection-style homogeneous operators next
2. Add a propagation-oriented homogeneous batch with minimal overlap
3. Pause operator work and refactor `MessagePassing` to absorb more variants

The chosen direction is:

> Add a propagation-oriented homogeneous batch with minimal overlap and keep implementation logic local to the new layers.

This gives `vgl` broader operator coverage without duplicating the semantics already covered by `GCNConv`, `GINConv`, `TAGConv`, `SGConv`, or `ChebConv`. A `MessagePassing` refactor would increase internal churn before there is enough pressure to justify it.

## Architecture

Phase 8 should extend `vgl.nn.conv` with:

- `AGNNConv`
- `LightGCNConv`
- `FAGCNConv`

These layers should live under:

- `vgl/nn/conv/agnn.py`
- `vgl/nn/conv/lightgcn.py`
- `vgl/nn/conv/fagcn.py`

They should be exported from:

- `vgl.nn.conv`
- `vgl.nn`
- `vgl`

The package should continue exposing one flat built-in operator surface. No `recommendation`, `propagation`, or `experimental` namespace should be introduced.

## Public API Shape

The user-facing import path should be:

```python
from vgl.nn.conv import AGNNConv, LightGCNConv, FAGCNConv
```

and also:

```python
from vgl import AGNNConv, LightGCNConv, FAGCNConv
```

The constructors should be:

```python
conv = AGNNConv(channels=64, beta=1.0, train_beta=False)
out = conv(graph)

conv = LightGCNConv()
out = conv(graph)

conv = FAGCNConv(channels=64, eps=0.1)
out = conv(graph)
```

This is intentionally different from projection-style layers with `in_channels` and `out_channels`. These operators act as propagation modules that preserve feature width.

## Operator Semantics

### Shared propagation contract

All three operators should:

- support homogeneous graphs only
- preserve feature width
- preserve input dtype and device
- support `conv(graph)` and `conv(x, edge_index)`
- fail early on heterogeneous graphs with `"homogeneous"` in the error message

They should fit into models like:

```python
x = input_proj(graph.x)
x = conv(x, graph.edge_index)
logits = head(x)
```

### AGNNConv

`AGNNConv` should expose:

- `channels`
- `beta=1.0`
- `train_beta=False`

The semantic should be:

- compute pairwise edge attention from normalized source and target node features
- scale attention logits by `beta`
- normalize incoming edge weights per destination node
- propagate weighted source features to the destination nodes

Output shape:

- `[num_nodes, channels]`

The first version should not include:

- multi-head attention
- edge features
- dropout
- residual toggles

### LightGCNConv

`LightGCNConv` should expose no learned projection parameters:

- `LightGCNConv()`

The semantic should be:

- run one normalized propagation step
- return the propagated representation directly

Output shape:

- `[num_nodes, channels]`

This keeps the operator faithful to the "light" propagation idea. Layer weighting across depths belongs in user models or later phases, not in the first primitive.

### FAGCNConv

`FAGCNConv` should expose:

- `channels`
- `eps=0.1`

The semantic should be:

- compute one normalized neighbor propagation
- compute a learned gate from source and target features
- rescale propagated messages with the gate
- mix the gated propagation with an `eps`-scaled identity path

Output shape:

- `[num_nodes, channels]`

The first version should not include:

- configurable normalization families
- separate residual matrices
- dropout switches

## Internal Helper Boundary

Phase 8 should continue using `vgl/nn/conv/_homo.py` for tiny shared helpers only.

It is acceptable to add helpers such as:

- destination-degree normalization
- edge-level softmax over incoming edges
- attention-weighted aggregation

It is not acceptable to use this phase as an excuse to redesign `MessagePassing`.

## Testing Strategy

Per approved scope, Phase 8 should use a compact testing surface rather than a full regression campaign.

### Contract tests

In `tests/nn/test_convs.py` add:

- `AGNNConv` on `Graph`
- `AGNNConv` on `(x, edge_index)`
- `LightGCNConv` on `Graph`
- `LightGCNConv` on `(x, edge_index)`
- `FAGCNConv` on `Graph`
- heterogeneous graph rejection tests for all three

### Export tests

`tests/test_package_exports.py` should assert that:

- `AGNNConv`
- `LightGCNConv`
- `FAGCNConv`

are exposed from the root package.

### Compact integration path

Extend `tests/integration/test_homo_conv_zoo.py` so these operators also run through the current homogeneous training loop.

### Example surface

Extend `examples/homo/conv_zoo.py` to include:

- `agnn`
- `lightgcn`
- `fagcn`

## Verification Boundary

Phase 8 should verify only the newly touched operator surface:

- focused operator and export tests
- compact training-loop integration test
- `conv_zoo.py` smoke run

This phase should not require:

- full `pytest -v`
- `ruff check`
- `mypy`
- full example smoke suite

Those broader checks can happen in a later stabilization pass.

## Deliverables

Phase 8 should ship:

- `AGNNConv`
- `LightGCNConv`
- `FAGCNConv`
- exports from `vgl.nn.conv`, `vgl.nn`, and `vgl`
- focused contract tests
- compact integration coverage
- an expanded `conv_zoo` example
- minimal doc updates where needed

## Explicit Non-Goals

Do not include:

- heterogeneous operator work
- temporal operator work
- recommendation-specific loss or ranking utilities
- layer-weight aggregation containers for LightGCN
- a `MessagePassing` rewrite
- full regression verification for the repository

## Repository Touchpoints

Phase 8 will mostly affect:

- `vgl/nn/conv/`
- `vgl/nn/__init__.py`
- `vgl/__init__.py`
- `tests/nn/`
- `tests/integration/`
- `tests/test_package_exports.py`
- `examples/homo/conv_zoo.py`
- `README.md`
- `docs/`

## Acceptance Criteria

Phase 8 is complete when:

1. `AGNNConv`, `LightGCNConv`, and `FAGCNConv` are publicly exported
2. all three operators work on homogeneous graphs and support both call styles
3. the focused tests for exports, shapes, and hetero rejection pass
4. the compact integration test passes with the new operators
5. `conv_zoo.py` demonstrates the expanded operator set
6. no commits are made until explicit user approval

## Next Step

The next step is to write an implementation plan with exact file edits, test commands, and no-commit execution checkpoints.
