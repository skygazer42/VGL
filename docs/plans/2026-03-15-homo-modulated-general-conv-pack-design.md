# Homogeneous Modulated and General Convolution Pack Design

**Date:** 2026-03-15
**Status:** Approved for planning

## Goal

Expand `vgl` with one more compact homogeneous operator batch by adding `GENConv`, `FiLMConv`, and `SimpleConv` as stable public layers that fit the current graph and training abstractions.

## Scope Decisions

- Focus on homogeneous graph operators only
- Add one compact batch:
  - `GENConv`
  - `FiLMConv`
  - `SimpleConv`
- Keep the public API aligned with current `vgl` conv usage patterns
- Continue supporting both `conv(graph)` and `conv(x, edge_index)`
- Allow `GENConv` and `FiLMConv` to remain projection-style layers
- Allow `SimpleConv` to remain a parameter-free equal-width propagation primitive
- Reuse the current training loop, export surface, and `conv_zoo` example

## Chosen Direction

Three directions were considered:

1. Keep adding only ultra-light propagation or aggregation layers
2. Add a medium-complexity batch centered on generalized aggregation and feature modulation
3. Jump directly to heavier layers such as `TransformerConv` or `PNAConv`

The chosen direction is:

> Add a medium-complexity batch centered on generalized aggregation and feature modulation.

This raises the representational surface of `vgl` without pushing the helper boundary as far as heavier attention or scaler-heavy operators would.

## Architecture

Phase 15 should extend `vgl.nn.conv` with:

- `GENConv`
- `FiLMConv`
- `SimpleConv`

These layers should live under:

- `vgl/nn/conv/gen.py`
- `vgl/nn/conv/film.py`
- `vgl/nn/conv/simple.py`

They should be exported from:

- `vgl.nn.conv`
- `vgl.nn`
- `vgl`

The built-in operator surface should remain flat. No new `general`, `modulated`, or `experimental` namespace should be introduced.

## Public API Shape

The user-facing import path should be:

```python
from vgl.nn.conv import GENConv, FiLMConv, SimpleConv
```

and also:

```python
from vgl import GENConv, FiLMConv, SimpleConv
```

The constructors should be:

```python
conv = GENConv(in_channels=64, out_channels=32, aggr="softmax", beta=1.0)
out = conv(graph)

conv = FiLMConv(in_channels=64, out_channels=32)
out = conv(graph)

conv = SimpleConv(aggr="mean")
out = conv(graph)
```

## Operator Semantics

### Shared contract

All three operators should:

- support homogeneous graphs only
- preserve input dtype and device
- support `conv(graph)` and `conv(x, edge_index)`
- fail early on heterogeneous graphs with `"homogeneous"` in the error message

### GENConv

`GENConv` should expose:

- `in_channels`
- `out_channels`
- `aggr="softmax"`
- `beta=1.0`

The first version should:

- project source-node features to message values
- support a compact aggregation family:
  - `softmax`
  - `mean`
  - `sum`
- for `softmax`, compute destination-normalized edge weights from message scores
- aggregate projected messages to destination nodes
- add one root-node projection branch

Output shape:

- `[num_nodes, out_channels]`

The first version should not include:

- edge features
- power mean aggregation
- message normalization modules
- full PyG parameter parity

### FiLMConv

`FiLMConv` should expose:

- `in_channels`
- `out_channels`

The first version should:

- project source-node features to message values
- generate feature-wise `gamma` and `beta` from destination-node features
- modulate each incoming message as `gamma * message + beta`
- aggregate the modulated messages to destination nodes
- add one root-node projection branch

Output shape:

- `[num_nodes, out_channels]`

The first version should not include:

- relation-specific conditioning
- edge features
- multi-head structure

### SimpleConv

`SimpleConv` should expose:

- `aggr="mean"`

The first version should:

- preserve feature width
- perform one pure neighborhood aggregation step
- support a compact aggregation family:
  - `mean`
  - `sum`
  - `max`

Output shape:

- `[num_nodes, channels]`

The first version should not include:

- root weighting
- learnable projections
- edge weights

## Internal Helper Boundary

Phase 15 should continue using `vgl/nn/conv/_homo.py` for tiny shared helpers only.

Acceptable additions:

- reuse of `coerce_homo_inputs(...)`
- reuse of `sum_propagate(...)`, `mean_propagate(...)`, `max_propagate(...)`, and `edge_softmax(...)`

Unacceptable additions:

- `MessagePassing` redesign
- a generalized modulation framework
- a large aggregation-runtime abstraction

## Testing Strategy

Per approved scope, Phase 15 should use a compact testing surface.

### Contract tests

In `tests/nn/test_convs.py` add:

- `GENConv` on `Graph`
- `GENConv` on `(x, edge_index)`
- `FiLMConv` on `Graph`
- `FiLMConv` on `(x, edge_index)`
- `SimpleConv` on `Graph`
- `SimpleConv` on `(x, edge_index)`
- heterogeneous graph rejection tests for all three

### Export tests

`tests/test_package_exports.py` should assert that:

- `GENConv`
- `FiLMConv`
- `SimpleConv`

are exposed from the root package.

### Compact integration path

Extend `tests/integration/test_homo_conv_zoo.py` so all three operators also run through the current homogeneous training loop.

### Example surface

Extend `examples/homo/conv_zoo.py` to include:

- `gen`
- `film`
- `simple`

## Verification Boundary

Phase 15 should verify only the touched operator surface:

- focused operator and export tests
- compact training-loop integration test
- `conv_zoo.py` smoke run

This phase should not require:

- full `pytest -v`
- `ruff check`
- `mypy`
- full example smoke suite

## Deliverables

Phase 15 should ship:

- `GENConv`
- `FiLMConv`
- `SimpleConv`
- exports from `vgl.nn.conv`, `vgl.nn`, and `vgl`
- focused contract tests
- compact integration coverage
- an expanded `conv_zoo` example
- minimal doc updates where needed

## Explicit Non-Goals

Do not include:

- heterogeneous operator work
- temporal operator work
- `TransformerConv`
- `PNAConv`
- `MessagePassing` refactor
- full regression verification for the repository

## Repository Touchpoints

Phase 15 will mostly affect:

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

Phase 15 is complete when:

1. `GENConv`, `FiLMConv`, and `SimpleConv` are publicly exported
2. all three operators work on homogeneous graphs and support both call styles
3. the focused tests for exports, shapes, and hetero rejection pass
4. the compact integration test passes with the new operators
5. `conv_zoo.py` demonstrates the expanded operator set
6. no commits are made until explicit user approval

## Next Step

The next step is to write an implementation plan with exact file edits, focused test commands, and no-commit execution checkpoints.
