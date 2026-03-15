# Homogeneous Residual and Basic Convolution Pack Design

**Date:** 2026-03-15
**Status:** Approved for planning

## Goal

Expand `vgl` with one more compact homogeneous operator batch by adding `GCN2Conv`, `GraphConv`, and `LEConv` as stable public layers that fit the current graph and training abstractions.

## Scope Decisions

- Focus on homogeneous graph operators only
- Add one compact batch:
  - `GCN2Conv`
  - `GraphConv`
  - `LEConv`
- Keep the public API aligned with current `vgl` conv usage patterns
- Continue supporting both `conv(graph)` and `conv(x, edge_index)`
- Allow `GCN2Conv` to remain equal-width propagation-oriented
- Allow `GraphConv` and `LEConv` to remain projection-style layers
- Reuse the current training loop, export surface, and `conv_zoo` example

## Chosen Direction

Three directions were considered:

1. Add a heavier mixed batch such as `GENConv` or `PNAConv`
2. Add a low-coupling batch centered on residual propagation and basic neighborhood aggregation
3. Pause operator work and redesign `MessagePassing` for broader operator coverage

The chosen direction is:

> Add a low-coupling batch centered on residual propagation and basic neighborhood aggregation.

This broadens the built-in operator surface without forcing a new abstraction layer. Heavier operators like `PNAConv` and `GENConv` would either require a much wider helper surface or a first-version semantic compromise that is not worth the churn right now.

## Architecture

Phase 13 should extend `vgl.nn.conv` with:

- `GCN2Conv`
- `GraphConv`
- `LEConv`

These layers should live under:

- `vgl/nn/conv/gcn2.py`
- `vgl/nn/conv/graphconv.py`
- `vgl/nn/conv/leconv.py`

They should be exported from:

- `vgl.nn.conv`
- `vgl.nn`
- `vgl`

The built-in operator surface should remain flat. No new `advanced`, `residual`, `local`, or `experimental` namespace should be introduced.

## Public API Shape

The user-facing import path should be:

```python
from vgl.nn.conv import GCN2Conv, GraphConv, LEConv
```

and also:

```python
from vgl import GCN2Conv, GraphConv, LEConv
```

The constructors should be:

```python
conv = GCN2Conv(channels=64, alpha=0.1, theta=1.0, layer=1, shared_weights=True)
out = conv(graph, x0=graph.x)

conv = GraphConv(in_channels=64, out_channels=32)
out = conv(graph)

conv = LEConv(in_channels=64, out_channels=32)
out = conv(graph)
```

## Operator Semantics

### Shared contract

All three operators should:

- support homogeneous graphs only
- preserve input dtype and device
- support `conv(graph)` and `conv(x, edge_index)`
- fail early on heterogeneous graphs with `"homogeneous"` in the error message

### GCN2Conv

`GCN2Conv` should expose:

- `channels`
- `alpha=0.1`
- `theta=1.0`
- `layer=1`
- `shared_weights=True`

The first version should:

- preserve feature width
- run one normalized propagation step
- mix the propagated state with an initial representation `x0` through `alpha`
- apply one residual-style linear transform controlled by `theta` and `layer`
- accept `x0` explicitly in both call styles

Output shape:

- `[num_nodes, channels]`

The first version should not include:

- deeper stacked containers
- dropout
- batch norm
- full PyG parameter parity

### GraphConv

`GraphConv` should expose:

- `in_channels`
- `out_channels`

The first version should:

- compute a neighborhood sum aggregation
- compute a root-node linear branch
- add the two branches together

Output shape:

- `[num_nodes, out_channels]`

The first version should not include:

- aggregator selection
- edge weights
- normalization toggles

### LEConv

`LEConv` should expose:

- `in_channels`
- `out_channels`

The first version should:

- project source, destination, and self features
- build edge messages from local source-destination differences
- aggregate those messages to destination nodes
- add a self branch

Output shape:

- `[num_nodes, out_channels]`

The first version should not include:

- edge features
- attention
- extra normalization families

## Internal Helper Boundary

Phase 13 should continue using `vgl/nn/conv/_homo.py` for tiny shared helpers only.

Acceptable additions:

- a tiny `resolve_optional_reference(...)` helper for `GCN2Conv`
- reuse of `sum_propagate(...)` and `symmetric_propagate(...)`

Unacceptable additions:

- `MessagePassing` redesign
- a generalized edge-program runtime
- a new multi-operator abstraction layer

## Testing Strategy

Per approved scope, Phase 13 should use a compact testing surface.

### Contract tests

In `tests/nn/test_convs.py` add:

- `GCN2Conv` on `Graph` with explicit `x0`
- `GCN2Conv` on `(x, edge_index)` with explicit `x0`
- `GraphConv` on `Graph`
- `GraphConv` on `(x, edge_index)`
- `LEConv` on `Graph`
- `LEConv` on `(x, edge_index)`
- heterogeneous graph rejection tests for all three

### Export tests

`tests/test_package_exports.py` should assert that:

- `GCN2Conv`
- `GraphConv`
- `LEConv`

are exposed from the root package.

### Compact integration path

Extend `tests/integration/test_homo_conv_zoo.py` so all three operators also run through the current homogeneous training loop.

For `GCN2Conv`, the tiny integration model should explicitly pass `graph.x` as `x0`.

### Example surface

Extend `examples/homo/conv_zoo.py` to include:

- `gcn2`
- `graphconv`
- `leconv`

## Verification Boundary

Phase 13 should verify only the touched operator surface:

- focused operator and export tests
- compact training-loop integration test
- `conv_zoo.py` smoke run

This phase should not require:

- full `pytest -v`
- `ruff check`
- `mypy`
- full example smoke suite

## Deliverables

Phase 13 should ship:

- `GCN2Conv`
- `GraphConv`
- `LEConv`
- exports from `vgl.nn.conv`, `vgl.nn`, and `vgl`
- focused contract tests
- compact integration coverage
- an expanded `conv_zoo` example
- minimal doc updates where needed

## Explicit Non-Goals

Do not include:

- heterogeneous operator work
- temporal operator work
- `PNAConv`, `GENConv`, or `ClusterGCNConv`
- `MessagePassing` refactor
- full regression verification for the repository

## Repository Touchpoints

Phase 13 will mostly affect:

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

Phase 13 is complete when:

1. `GCN2Conv`, `GraphConv`, and `LEConv` are publicly exported
2. all three operators work on homogeneous graphs and support both call styles
3. `GCN2Conv` accepts an explicit initial representation `x0`
4. the focused tests for exports, shapes, and hetero rejection pass
5. the compact integration test passes with the new operators
6. `conv_zoo.py` demonstrates the expanded operator set
7. no commits are made until explicit user approval

## Next Step

The next step is to write an implementation plan with exact file edits, focused test commands, and no-commit execution checkpoints.
