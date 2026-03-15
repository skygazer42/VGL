# Homogeneous Gated and Cluster Convolution Pack Design

**Date:** 2026-03-15
**Status:** Approved for planning

## Goal

Expand `vgl` with one more compact homogeneous operator batch by adding `ResGatedGraphConv`, `GatedGraphConv`, and `ClusterGCNConv` as stable public layers that fit the current graph and training abstractions.

## Scope Decisions

- Focus on homogeneous graph operators only
- Add one compact batch:
  - `ResGatedGraphConv`
  - `GatedGraphConv`
  - `ClusterGCNConv`
- Keep the public API aligned with current `vgl` conv usage patterns
- Continue supporting both `conv(graph)` and `conv(x, edge_index)`
- Allow `GatedGraphConv` to remain an equal-width recurrent propagation layer
- Allow `ResGatedGraphConv` and `ClusterGCNConv` to remain projection-style layers
- Reuse the current training loop, export surface, and `conv_zoo` example

## Chosen Direction

Three directions were considered:

1. Add a heavier batch such as `PNAConv` or `TransformerConv`
2. Add a low-coupling gated and cluster-oriented batch with minimal helper growth
3. Pause operator work and run a broad internal unification pass first

The chosen direction is:

> Add a low-coupling gated and cluster-oriented batch with minimal helper growth.

This expands the built-in operator surface into gated message passing and cluster-style propagation without forcing a new abstraction layer. Heavier operators would widen the helper surface too aggressively for the current stability-first phase.

## Architecture

Phase 14 should extend `vgl.nn.conv` with:

- `ResGatedGraphConv`
- `GatedGraphConv`
- `ClusterGCNConv`

These layers should live under:

- `vgl/nn/conv/resgated.py`
- `vgl/nn/conv/gatedgraph.py`
- `vgl/nn/conv/clustergcn.py`

They should be exported from:

- `vgl.nn.conv`
- `vgl.nn`
- `vgl`

The built-in operator surface should remain flat. No new `gated`, `recurrent`, `cluster`, or `experimental` namespace should be introduced.

## Public API Shape

The user-facing import path should be:

```python
from vgl.nn.conv import ResGatedGraphConv, GatedGraphConv, ClusterGCNConv
```

and also:

```python
from vgl import ResGatedGraphConv, GatedGraphConv, ClusterGCNConv
```

The constructors should be:

```python
conv = ResGatedGraphConv(in_channels=64, out_channels=32)
out = conv(graph)

conv = GatedGraphConv(channels=64, steps=3)
out = conv(graph)

conv = ClusterGCNConv(in_channels=64, out_channels=32, diag_lambda=0.0)
out = conv(graph)
```

## Operator Semantics

### Shared contract

All three operators should:

- support homogeneous graphs only
- preserve input dtype and device
- support `conv(graph)` and `conv(x, edge_index)`
- fail early on heterogeneous graphs with `"homogeneous"` in the error message

### ResGatedGraphConv

`ResGatedGraphConv` should expose:

- `in_channels`
- `out_channels`

The first version should:

- project source and destination features to an output-width gate space
- compute a sigmoid gate from source and destination projections
- project source features to message values
- aggregate gated source messages to destination nodes
- add one root-node projection branch

Output shape:

- `[num_nodes, out_channels]`

The first version should not include:

- edge features
- multi-head structure
- attention dropout

### GatedGraphConv

`GatedGraphConv` should expose:

- `channels`
- `steps=3`

The first version should:

- preserve feature width
- require the input width to match `channels`
- iteratively propagate transformed node states
- update states with one `GRUCell`

Output shape:

- `[num_nodes, channels]`

The first version should not include:

- input projection from arbitrary widths
- custom aggregation families
- edge-conditioned recurrent messages

### ClusterGCNConv

`ClusterGCNConv` should expose:

- `in_channels`
- `out_channels`
- `diag_lambda=0.0`

The first version should:

- compute one neighborhood aggregation
- apply one neighborhood projection branch
- apply one root-node projection branch scaled by `1 + diag_lambda`
- return the sum of both branches

Output shape:

- `[num_nodes, out_channels]`

The first version should not include:

- explicit partition metadata
- cached normalization
- sparse-cluster training utilities

## Internal Helper Boundary

Phase 14 should continue using `vgl/nn/conv/_homo.py` for tiny shared helpers only.

Acceptable additions:

- reuse of `coerce_homo_inputs(...)`
- reuse of `sum_propagate(...)` and `mean_propagate(...)`

Unacceptable additions:

- `MessagePassing` redesign
- a generalized gated-edge runtime
- a multi-operator recurrent framework

## Testing Strategy

Per approved scope, Phase 14 should use a compact testing surface.

### Contract tests

In `tests/nn/test_convs.py` add:

- `ResGatedGraphConv` on `Graph`
- `ResGatedGraphConv` on `(x, edge_index)`
- `GatedGraphConv` on `Graph`
- `GatedGraphConv` on `(x, edge_index)`
- `ClusterGCNConv` on `Graph`
- `ClusterGCNConv` on `(x, edge_index)`
- heterogeneous graph rejection tests for all three

### Export tests

`tests/test_package_exports.py` should assert that:

- `ResGatedGraphConv`
- `GatedGraphConv`
- `ClusterGCNConv`

are exposed from the root package.

### Compact integration path

Extend `tests/integration/test_homo_conv_zoo.py` so all three operators also run through the current homogeneous training loop.

### Example surface

Extend `examples/homo/conv_zoo.py` to include:

- `resgated`
- `gatedgraph`
- `clustergcn`

## Verification Boundary

Phase 14 should verify only the touched operator surface:

- focused operator and export tests
- compact training-loop integration test
- `conv_zoo.py` smoke run

This phase should not require:

- full `pytest -v`
- `ruff check`
- `mypy`
- full example smoke suite

## Deliverables

Phase 14 should ship:

- `ResGatedGraphConv`
- `GatedGraphConv`
- `ClusterGCNConv`
- exports from `vgl.nn.conv`, `vgl.nn`, and `vgl`
- focused contract tests
- compact integration coverage
- an expanded `conv_zoo` example
- minimal doc updates where needed

## Explicit Non-Goals

Do not include:

- heterogeneous operator work
- temporal operator work
- `PNAConv`
- `TransformerConv`
- `MessagePassing` refactor
- full regression verification for the repository

## Repository Touchpoints

Phase 14 will mostly affect:

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

Phase 14 is complete when:

1. `ResGatedGraphConv`, `GatedGraphConv`, and `ClusterGCNConv` are publicly exported
2. all three operators work on homogeneous graphs and support both call styles
3. the focused tests for exports, shapes, and hetero rejection pass
4. the compact integration test passes with the new operators
5. `conv_zoo.py` demonstrates the expanded operator set
6. no commits are made until explicit user approval

## Next Step

The next step is to write an implementation plan with exact file edits, focused test commands, and no-commit execution checkpoints.
