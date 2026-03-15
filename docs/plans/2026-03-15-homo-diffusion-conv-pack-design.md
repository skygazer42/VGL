# Homogeneous Diffusion Convolution Pack Design

**Date:** 2026-03-15
**Status:** Approved for planning

## Goal

Expand `vgl` with a new homogeneous diffusion-style operator batch by adding `SSGConv` and `DAGNNConv` as stable public convolution layers that fit the current graph and training abstractions.

## Scope Decisions

- Focus on homogeneous graph operators only
- Add one compact diffusion batch:
  - `SSGConv`
  - `DAGNNConv`
- Keep the public API aligned with current `vgl` conv usage patterns
- Continue supporting both `conv(graph)` and `conv(x, edge_index)`
- Keep both operators equal-width propagation layers
- Reuse the existing training loop and `conv_zoo` example surface

## Chosen Direction

Three directions were considered:

1. Implement a full `DAGNN` model instead of a public conv layer
2. Expose `SSGConv` and `DAGNNConv` as stable public conv modules
3. Delay diffusion work and continue only with polynomial and recursive propagation layers

The chosen direction is:

> Expose `SSGConv` and `DAGNNConv` as stable public conv modules and keep the training path unchanged.

This keeps the package on the "more convs first" trajectory and avoids introducing a one-off model abstraction that would not generalize cleanly across the existing operator surface.

## Architecture

Phase 11 should extend `vgl.nn.conv` with:

- `SSGConv`
- `DAGNNConv`

These layers should live under:

- `vgl/nn/conv/ssg.py`
- `vgl/nn/conv/dagnn.py`

They should be exported from:

- `vgl.nn.conv`
- `vgl.nn`
- `vgl`

The built-in operator surface should remain flat. No new `diffusion`, `deep`, or `experimental` namespace should be introduced.

## Public API Shape

The user-facing import path should be:

```python
from vgl.nn.conv import SSGConv, DAGNNConv
```

and also:

```python
from vgl import SSGConv, DAGNNConv
```

The constructors should be:

```python
conv = SSGConv(channels=64, steps=10, alpha=0.1)
out = conv(graph)

conv = DAGNNConv(channels=64, steps=3)
out = conv(graph)
```

Both APIs should preserve feature width and avoid large option matrices.

## Operator Semantics

### Shared propagation contract

Both operators should:

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

### SSGConv

`SSGConv` should expose:

- `channels`
- `steps=10`
- `alpha=0.1`

The first version should:

- build propagation outputs from order `0` through `steps`
- compute the mean of the propagated orders excluding the original input
- mix that mean with the original features through `alpha`

Output shape:

- `[num_nodes, channels]`

The first version should not include:

- alternative normalization families
- cache controls
- paper-level parameter parity

### DAGNNConv

`DAGNNConv` should expose:

- `channels`
- `steps=3`

The first version should:

- build propagation outputs from order `0` through `steps`
- use a small learned gate to score each propagation order per node
- normalize those order scores across depth
- return the weighted sum of the order representations

Output shape:

- `[num_nodes, channels]`

The first version should not include:

- a full public backbone class
- extra dropout or residual switches
- multi-gate variants

## Internal Helper Boundary

Phase 11 should continue using `vgl/nn/conv/_homo.py` for tiny shared helpers only.

Acceptable additions:

- reuse of `propagate_steps(...)`
- tiny depth-softmax or gating helper if genuinely needed

Unacceptable additions:

- `MessagePassing` redesign
- a generalized depth-composition framework

## Testing Strategy

Per approved scope, Phase 11 should use a compact testing surface.

### Contract tests

In `tests/nn/test_convs.py` add:

- `SSGConv` on `Graph`
- `SSGConv` on `(x, edge_index)`
- `DAGNNConv` on `Graph`
- `DAGNNConv` on `(x, edge_index)`
- heterogeneous graph rejection tests for both operators

### Export tests

`tests/test_package_exports.py` should assert that:

- `SSGConv`
- `DAGNNConv`

are exposed from the root package.

### Compact integration path

Extend `tests/integration/test_homo_conv_zoo.py` so both operators also run through the current homogeneous training loop.

### Example surface

Extend `examples/homo/conv_zoo.py` to include:

- `ssg`
- `dagnn`

## Verification Boundary

Phase 11 should verify only the touched operator surface:

- focused operator and export tests
- compact training-loop integration test
- `conv_zoo.py` smoke run

This phase should not require:

- full `pytest -v`
- `ruff check`
- `mypy`
- full example smoke suite

## Deliverables

Phase 11 should ship:

- `SSGConv`
- `DAGNNConv`
- exports from `vgl.nn.conv`, `vgl.nn`, and `vgl`
- focused contract tests
- compact integration coverage
- an expanded `conv_zoo` example
- minimal doc updates where needed

## Explicit Non-Goals

Do not include:

- heterogeneous operator work
- temporal operator work
- a `DAGNNModel` public class
- extra recommendation or ranking utilities
- `MessagePassing` refactor
- full regression verification for the repository

## Repository Touchpoints

Phase 11 will mostly affect:

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

Phase 11 is complete when:

1. `SSGConv` and `DAGNNConv` are publicly exported
2. both operators work on homogeneous graphs and support both call styles
3. both operators preserve feature width
4. the focused tests for exports, shapes, and hetero rejection pass
5. the compact integration test passes with the new operators
6. `conv_zoo.py` demonstrates the expanded operator set
7. no commits are made until explicit user approval

## Next Step

The next step is to write an implementation plan with exact file edits, focused test commands, and no-commit execution checkpoints.
