# Homogeneous Polynomial Filter Convolution Pack Design

**Date:** 2026-03-15
**Status:** Approved for planning

## Goal

Expand `vgl` with a new homogeneous polynomial-filter batch by adding `MixHopConv` and `BernConv` as stable public operators that fit the current graph and training abstractions.

## Scope Decisions

- Focus on homogeneous graph operators only
- Add one compact polynomial-filter batch:
  - `MixHopConv`
  - `BernConv`
- Keep the public API aligned with current `vgl` conv usage patterns
- Continue supporting both `conv(graph)` and `conv(x, edge_index)`
- Allow `MixHopConv` to be a projection-style layer
- Keep `BernConv` as an equal-width propagation layer
- Reuse the existing training loop and `conv_zoo` example surface

## Chosen Direction

Three directions were considered:

1. Force both operators into equal-width propagation APIs
2. Let `MixHopConv` be projection-style and keep `BernConv` equal-width
3. Delay polynomial filters and continue only with recursive propagation layers

The chosen direction is:

> Let `MixHopConv` keep its natural multi-hop projection semantics and keep `BernConv` as an equal-width propagation layer.

This gives `vgl` broader operator coverage without distorting either layer to fit an artificial common interface.

## Architecture

Phase 10 should extend `vgl.nn.conv` with:

- `MixHopConv`
- `BernConv`

These layers should live under:

- `vgl/nn/conv/mixhop.py`
- `vgl/nn/conv/bern.py`

They should be exported from:

- `vgl.nn.conv`
- `vgl.nn`
- `vgl`

The operator surface should remain flat. No `polynomial`, `advanced`, or `experimental` namespace should be introduced.

## Public API Shape

The user-facing import path should be:

```python
from vgl.nn.conv import MixHopConv, BernConv
```

and also:

```python
from vgl import MixHopConv, BernConv
```

The constructors should be:

```python
conv = MixHopConv(in_channels=64, out_channels=32, powers=(0, 1, 2))
out = conv(graph)

conv = BernConv(channels=64, steps=3)
out = conv(graph)
```

## Operator Semantics

### Shared propagation contract

Both operators should:

- support homogeneous graphs only
- preserve input dtype and device
- support `conv(graph)` and `conv(x, edge_index)`
- fail early on heterogeneous graphs with `"homogeneous"` in the error message

### MixHopConv

`MixHopConv` should expose:

- `in_channels`
- `out_channels`
- `powers=(0, 1, 2)`

The first version should:

- compute propagation outputs for the requested powers
- use `A^0 X` as the original features
- use repeated normalized propagation for higher powers
- concatenate the selected power outputs
- apply one linear projection to `out_channels`

Output shape:

- `[num_nodes, out_channels]`

The first version should not include:

- per-power separate linear layers
- dropout
- normalization option matrices
- paper-level full parameter parity

### BernConv

`BernConv` should expose:

- `channels`
- `steps=3`

The first version should:

- build propagation outputs from order `0` through `steps`
- initialize learnable order coefficients from normalized binomial weights
- return the weighted sum of those propagation outputs

Output shape:

- `[num_nodes, channels]`

The first version should not include:

- alternative polynomial family switches
- explicit regularization interfaces
- separate propagation primitive exposure

## Internal Helper Boundary

Phase 10 should continue using `vgl/nn/conv/_homo.py` for tiny shared helpers only.

Acceptable additions:

- reuse of `propagate_steps(...)`
- tiny helper for binomial initialization if needed

Unacceptable additions:

- `MessagePassing` redesign
- a meta-framework to unify all conv families

## Testing Strategy

Per approved scope, Phase 10 should use a compact testing surface.

### Contract tests

In `tests/nn/test_convs.py` add:

- `MixHopConv` on `Graph`
- `MixHopConv` on `(x, edge_index)`
- `BernConv` on `Graph`
- `BernConv` on `(x, edge_index)`
- heterogeneous graph rejection tests for both operators

### Export tests

`tests/test_package_exports.py` should assert that:

- `MixHopConv`
- `BernConv`

are exposed from the root package.

### Compact integration path

Extend `tests/integration/test_homo_conv_zoo.py` so both operators also run through the current homogeneous training loop.

### Example surface

Extend `examples/homo/conv_zoo.py` to include:

- `mixhop`
- `bern`

## Verification Boundary

Phase 10 should verify only the touched operator surface:

- focused operator and export tests
- compact training-loop integration test
- `conv_zoo.py` smoke run

This phase should not require:

- full `pytest -v`
- `ruff check`
- `mypy`
- full example smoke suite

## Deliverables

Phase 10 should ship:

- `MixHopConv`
- `BernConv`
- exports from `vgl.nn.conv`, `vgl.nn`, and `vgl`
- focused contract tests
- compact integration coverage
- an expanded `conv_zoo` example
- minimal doc updates where needed

## Explicit Non-Goals

Do not include:

- heterogeneous operator work
- temporal operator work
- graph-level polynomial readout
- separate polynomial container abstractions
- `MessagePassing` refactor
- full regression verification for the repository

## Repository Touchpoints

Phase 10 will mostly affect:

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

Phase 10 is complete when:

1. `MixHopConv` and `BernConv` are publicly exported
2. both operators work on homogeneous graphs and support both call styles
3. `MixHopConv` returns `[N, out_channels]`
4. `BernConv` returns `[N, channels]`
5. the focused tests for exports, shapes, and hetero rejection pass
6. the compact integration test passes with the new operators
7. `conv_zoo.py` demonstrates the expanded operator set
8. no commits are made until explicit user approval

## Next Step

The next step is to write an implementation plan with exact file edits, focused test commands, and no-commit execution checkpoints.
