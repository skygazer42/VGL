# Homogeneous Structural Aggregation Convolution Pack Design

**Date:** 2026-03-15
**Status:** Approved for planning

## Goal

Expand `vgl` with a new homogeneous structural-aggregation operator batch by adding `H2GCNConv` and `EGConv` as stable public convolution layers that fit the current graph and training abstractions.

## Scope Decisions

- Focus on homogeneous graph operators only
- Add one compact structural-aggregation batch:
  - `H2GCNConv`
  - `EGConv`
- Keep the public API aligned with current `vgl` conv usage patterns
- Continue supporting both `conv(graph)` and `conv(x, edge_index)`
- Allow both operators to be projection-style layers
- Reuse the existing training loop and `conv_zoo` example surface

## Chosen Direction

Three directions were considered:

1. Force both operators into equal-width propagation APIs
2. Let both operators keep projection-style aggregation semantics
3. Delay this batch and continue only with propagation-family operators

The chosen direction is:

> Let both `H2GCNConv` and `EGConv` keep projection-style aggregation semantics.

This avoids distorting the operator meaning just to match earlier equal-width propagation layers.

## Architecture

Phase 12 should extend `vgl.nn.conv` with:

- `H2GCNConv`
- `EGConv`

These layers should live under:

- `vgl/nn/conv/h2gcn.py`
- `vgl/nn/conv/egconv.py`

They should be exported from:

- `vgl.nn.conv`
- `vgl.nn`
- `vgl`

The built-in operator surface should remain flat. No new `structural`, `advanced`, or `experimental` namespace should be introduced.

## Public API Shape

The user-facing import path should be:

```python
from vgl.nn.conv import H2GCNConv, EGConv
```

and also:

```python
from vgl import H2GCNConv, EGConv
```

The constructors should be:

```python
conv = H2GCNConv(in_channels=64, out_channels=32)
out = conv(graph)

conv = EGConv(in_channels=64, out_channels=32, aggregators=("sum", "mean", "max"))
out = conv(graph)
```

## Operator Semantics

### Shared propagation contract

Both operators should:

- support homogeneous graphs only
- preserve input dtype and device
- support `conv(graph)` and `conv(x, edge_index)`
- fail early on heterogeneous graphs with `"homogeneous"` in the error message

### H2GCNConv

`H2GCNConv` should expose:

- `in_channels`
- `out_channels`

The first version should:

- keep the original node features as the ego branch
- compute a one-hop neighborhood representation
- compute a two-hop neighborhood representation by propagating once more
- concatenate `ego`, `1-hop`, and `2-hop`
- apply one linear projection to `out_channels`

Output shape:

- `[num_nodes, out_channels]`

The first version should not include:

- explicit sparse matrix exclusion logic
- deeper backbone containers
- paper-level parameter parity

### EGConv

`EGConv` should expose:

- `in_channels`
- `out_channels`
- `aggregators=("sum", "mean", "max")`

The first version should:

- support exactly `sum`, `mean`, and `max`
- compute each requested aggregation over neighbors
- concatenate the requested aggregation outputs
- apply one linear projection to `out_channels`

Output shape:

- `[num_nodes, out_channels]`

The first version should not include:

- larger aggregator families
- multi-head routing
- edge feature branches

## Internal Helper Boundary

Phase 12 should continue using `vgl/nn/conv/_homo.py` for tiny shared helpers only.

Acceptable additions:

- tiny `sum` and `max` aggregation helpers if needed
- reuse of `mean_propagate(...)`

Unacceptable additions:

- `MessagePassing` redesign
- a generalized multi-aggregator framework for all convs

## Testing Strategy

Per approved scope, Phase 12 should use a compact testing surface.

### Contract tests

In `tests/nn/test_convs.py` add:

- `H2GCNConv` on `Graph`
- `H2GCNConv` on `(x, edge_index)`
- `EGConv` on `Graph`
- `EGConv` on `(x, edge_index)`
- heterogeneous graph rejection tests for both operators

### Export tests

`tests/test_package_exports.py` should assert that:

- `H2GCNConv`
- `EGConv`

are exposed from the root package.

### Compact integration path

Extend `tests/integration/test_homo_conv_zoo.py` so both operators also run through the current homogeneous training loop.

### Example surface

Extend `examples/homo/conv_zoo.py` to include:

- `h2gcn`
- `egconv`

## Verification Boundary

Phase 12 should verify only the touched operator surface:

- focused operator and export tests
- compact training-loop integration test
- `conv_zoo.py` smoke run

This phase should not require:

- full `pytest -v`
- `ruff check`
- `mypy`
- full example smoke suite

## Deliverables

Phase 12 should ship:

- `H2GCNConv`
- `EGConv`
- exports from `vgl.nn.conv`, `vgl.nn`, and `vgl`
- focused contract tests
- compact integration coverage
- an expanded `conv_zoo` example
- minimal doc updates where needed

## Explicit Non-Goals

Do not include:

- heterogeneous operator work
- temporal operator work
- deeper H2GCN containers
- additional EGConv aggregator families
- `MessagePassing` refactor
- full regression verification for the repository

## Repository Touchpoints

Phase 12 will mostly affect:

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

Phase 12 is complete when:

1. `H2GCNConv` and `EGConv` are publicly exported
2. both operators work on homogeneous graphs and support both call styles
3. both operators return `[num_nodes, out_channels]`
4. the focused tests for exports, shapes, and hetero rejection pass
5. the compact integration test passes with the new operators
6. `conv_zoo.py` demonstrates the expanded operator set
7. no commits are made until explicit user approval

## Next Step

The next step is to write an implementation plan with exact file edits, focused test commands, and no-commit execution checkpoints.
