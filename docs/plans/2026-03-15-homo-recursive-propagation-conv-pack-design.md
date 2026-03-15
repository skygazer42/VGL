# Homogeneous Recursive Propagation Convolution Pack Design

**Date:** 2026-03-15
**Status:** Approved for planning

## Goal

Expand `vgl` with a new homogeneous propagation batch by adding `ARMAConv` and `GPRGNNConv` as stable equal-width operators that fit the current graph and training abstractions.

## Scope Decisions

- Focus on homogeneous graph operators only
- Add one compact recursive propagation batch:
  - `ARMAConv`
  - `GPRGNNConv`
- Keep the public API aligned with current `vgl` conv usage patterns
- Continue supporting both `conv(graph)` and `conv(x, edge_index)`
- Treat both operators as equal-width propagation layers
- Reuse the existing training loop and `conv_zoo` example surface

## Chosen Direction

Three directions were considered:

1. Add only one recursive propagation operator next
2. Add a compact pair of recursive propagation operators with similar user ergonomics
3. Delay this batch and first redesign `MessagePassing`

The chosen direction is:

> Add a compact pair of recursive propagation operators with stable equal-width APIs and keep implementation logic local to the new layers.

This increases operator breadth without forcing another abstraction reset. It also complements the already added `APPNPConv`, `TAGConv`, `SGConv`, `ChebConv`, `AGNNConv`, `LightGCNConv`, and `FAGCNConv` with two more user-facing propagation styles.

## Architecture

Phase 9 should extend `vgl.nn.conv` with:

- `ARMAConv`
- `GPRGNNConv`

These layers should live under:

- `vgl/nn/conv/arma.py`
- `vgl/nn/conv/gprgnn.py`

They should be exported from:

- `vgl.nn.conv`
- `vgl.nn`
- `vgl`

The built-in operator surface should remain flat. No new `recursive`, `advanced`, or `experimental` namespace should be introduced.

## Public API Shape

The user-facing import path should be:

```python
from vgl.nn.conv import ARMAConv, GPRGNNConv
```

and also:

```python
from vgl import ARMAConv, GPRGNNConv
```

The constructors should be:

```python
conv = ARMAConv(channels=64, stacks=1, layers=2, alpha=0.1)
out = conv(graph)

conv = GPRGNNConv(channels=64, steps=10, alpha=0.1)
out = conv(graph)
```

These APIs intentionally preserve feature width and avoid large option matrices.

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

### ARMAConv

`ARMAConv` should expose:

- `channels`
- `stacks=1`
- `layers=2`
- `alpha=0.1`

The first version should:

- run `stacks` parallel recursive propagation chains
- run `layers` normalized propagation steps inside each chain
- mix each step with the initial representation through `alpha`
- return the average of all stack outputs

Output shape:

- `[num_nodes, channels]`

The first version should not include:

- dropout
- bias toggles
- full shared-weight option matrices
- paper-level parameter parity

### GPRGNNConv

`GPRGNNConv` should expose:

- `channels`
- `steps=10`
- `alpha=0.1`

The first version should:

- build representations from propagation order `0` through `steps`
- keep a learnable coefficient per propagation order
- initialize those coefficients from `alpha` in a stable, PPR-like way
- return the weighted sum of the order representations

Output shape:

- `[num_nodes, channels]`

The first version should not include:

- multiple initialization strategy names
- explicit regularization interfaces
- a separately exposed `GPRProp` primitive

## Internal Helper Boundary

Phase 9 should continue using `vgl/nn/conv/_homo.py` for tiny shared helpers only.

Acceptable additions:

- repeated normalized propagation helper
- simple propagation-order accumulation helper

Unacceptable additions:

- `MessagePassing` redesign
- generic meta-framework for all current convs

## Testing Strategy

Per approved scope, Phase 9 should use a compact testing surface.

### Contract tests

In `tests/nn/test_convs.py` add:

- `ARMAConv` on `Graph`
- `ARMAConv` on `(x, edge_index)`
- `GPRGNNConv` on `Graph`
- `GPRGNNConv` on `(x, edge_index)`
- heterogeneous graph rejection tests for both operators

### Export tests

`tests/test_package_exports.py` should assert that:

- `ARMAConv`
- `GPRGNNConv`

are exposed from the root package.

### Compact integration path

Extend `tests/integration/test_homo_conv_zoo.py` so both operators also run through the current homogeneous training loop.

### Example surface

Extend `examples/homo/conv_zoo.py` to include:

- `arma`
- `gprgnn`

## Verification Boundary

Phase 9 should verify only the touched operator surface:

- focused operator and export tests
- compact training-loop integration test
- `conv_zoo.py` smoke run

This phase should not require:

- full `pytest -v`
- `ruff check`
- `mypy`
- full example smoke suite

## Deliverables

Phase 9 should ship:

- `ARMAConv`
- `GPRGNNConv`
- exports from `vgl.nn.conv`, `vgl.nn`, and `vgl`
- focused contract tests
- compact integration coverage
- an expanded `conv_zoo` example
- minimal doc updates where needed

## Explicit Non-Goals

Do not include:

- heterogeneous operator work
- temporal operator work
- recommendation-specific ranking or loss utilities
- a separately exposed propagation container
- `MessagePassing` refactor
- full regression verification for the repository

## Repository Touchpoints

Phase 9 will mostly affect:

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

Phase 9 is complete when:

1. `ARMAConv` and `GPRGNNConv` are publicly exported
2. both operators work on homogeneous graphs and support both call styles
3. the focused tests for exports, shapes, and hetero rejection pass
4. the compact integration test passes with the new operators
5. `conv_zoo.py` demonstrates the expanded operator set
6. no commits are made until explicit user approval

## Next Step

The next step is to write an implementation plan with exact file edits, focused test commands, and no-commit execution checkpoints.
