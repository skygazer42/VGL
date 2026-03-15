# Homogeneous Local, Steered, and Degree Convolution Pack Design

**Date:** 2026-03-15
**Status:** Approved for planning

## Goal

Expand `vgl` with one more compact homogeneous operator batch by adding `EdgeConv`, `FeaStConv`, and `MFConv` as stable public layers that fit the current graph and training abstractions.

## Scope Decisions

- Focus on homogeneous graph operators only
- Add one compact batch:
  - `EdgeConv`
  - `FeaStConv`
  - `MFConv`
- Keep the public API aligned with current `vgl` conv usage patterns
- Continue supporting both `conv(graph)` and `conv(x, edge_index)`
- Keep all three layers independent from `edge_attr`
- Reuse the current training loop, export surface, and `conv_zoo` example

## Chosen Direction

Three directions were considered:

1. Add more general-purpose layers with overlap against `GENConv` and `GraphConv`
2. Add one batch centered on local pairwise interactions, feature-steered assignment, and degree-conditioned projection
3. Begin introducing `edge_attr`-aware layers now

The chosen direction is:

> Add one batch centered on local pairwise interactions, feature-steered assignment, and degree-conditioned projection.

This raises algorithm breadth without widening the graph data contract to edge attributes yet. It also avoids excessive overlap with `GENConv`, `GraphConv`, and `SimpleConv`.

## Architecture

Phase 16 should extend `vgl.nn.conv` with:

- `EdgeConv`
- `FeaStConv`
- `MFConv`

These layers should live under:

- `vgl/nn/conv/edgeconv.py`
- `vgl/nn/conv/feast.py`
- `vgl/nn/conv/mfconv.py`

They should be exported from:

- `vgl.nn.conv`
- `vgl.nn`
- `vgl`

The built-in operator surface should remain flat. No new `local`, `steered`, `dynamic`, or `experimental` namespace should be introduced.

## Public API Shape

The user-facing import path should be:

```python
from vgl.nn.conv import EdgeConv, FeaStConv, MFConv
```

and also:

```python
from vgl import EdgeConv, FeaStConv, MFConv
```

The constructors should be:

```python
conv = EdgeConv(in_channels=64, out_channels=32, aggr="max")
out = conv(graph)

conv = FeaStConv(in_channels=64, out_channels=32, heads=4)
out = conv(graph)

conv = MFConv(in_channels=64, out_channels=32, max_degree=4)
out = conv(graph)
```

## Operator Semantics

### Shared contract

All three operators should:

- support homogeneous graphs only
- preserve input dtype and device
- support `conv(graph)` and `conv(x, edge_index)`
- fail early on heterogeneous graphs with `"homogeneous"` in the error message

### EdgeConv

`EdgeConv` should expose:

- `in_channels`
- `out_channels`
- `aggr="max"`

The first version should:

- build edge messages from `[x_dst, x_src - x_dst]`
- apply an internal MLP to those pairwise edge features
- aggregate messages to destination nodes
- support a compact aggregation family:
  - `max`
  - `sum`
  - `mean`

Output shape:

- `[num_nodes, out_channels]`

The first version should not include:

- user-provided arbitrary `nn` modules
- edge features
- dynamic graph rebuilding

### FeaStConv

`FeaStConv` should expose:

- `in_channels`
- `out_channels`
- `heads=4`

The first version should:

- compute feature-steered assignment logits from `x_src - x_dst`
- normalize assignments per destination node
- project source features into `heads` value spaces
- aggregate head outputs and reduce them back to `[num_nodes, out_channels]`
- add one root-node projection branch

Output shape:

- `[num_nodes, out_channels]`

The first version should not include:

- edge features
- self-loop policy controls
- paper-level parameter parity

### MFConv

`MFConv` should expose:

- `in_channels`
- `out_channels`
- `max_degree=4`

The first version should:

- compute one neighborhood summary
- compute destination-node in-degree
- clip degree buckets to `max_degree`
- choose degree-specific linear transforms for neighborhood and root branches
- return the degree-conditioned sum

Output shape:

- `[num_nodes, out_channels]`

The first version should not include:

- custom degree statistics
- cached normalization
- relation-specific logic

## Internal Helper Boundary

Phase 16 should continue using `vgl/nn/conv/_homo.py` for tiny shared helpers only.

Acceptable additions:

- reuse of `coerce_homo_inputs(...)`
- reuse of `edge_softmax(...)`, `mean_propagate(...)`, and `sum_propagate(...)`

Unacceptable additions:

- `MessagePassing` redesign
- an edge-program runtime
- an edge-attribute data contract

## Testing Strategy

Per approved scope, Phase 16 should use a compact testing surface.

### Contract tests

In `tests/nn/test_convs.py` add:

- `EdgeConv` on `Graph`
- `EdgeConv` on `(x, edge_index)`
- `FeaStConv` on `Graph`
- `FeaStConv` on `(x, edge_index)`
- `MFConv` on `Graph`
- `MFConv` on `(x, edge_index)`
- heterogeneous graph rejection tests for all three

### Export tests

`tests/test_package_exports.py` should assert that:

- `EdgeConv`
- `FeaStConv`
- `MFConv`

are exposed from the root package.

### Compact integration path

Extend `tests/integration/test_homo_conv_zoo.py` so all three operators also run through the current homogeneous training loop.

### Example surface

Extend `examples/homo/conv_zoo.py` to include:

- `edgeconv`
- `feast`
- `mfconv`

## Verification Boundary

Phase 16 should verify only the touched operator surface:

- focused operator and export tests
- compact training-loop integration test
- `conv_zoo.py` smoke run

This phase should not require:

- full `pytest -v`
- `ruff check`
- `mypy`
- full example smoke suite

## Deliverables

Phase 16 should ship:

- `EdgeConv`
- `FeaStConv`
- `MFConv`
- exports from `vgl.nn.conv`, `vgl.nn`, and `vgl`
- focused contract tests
- compact integration coverage
- an expanded `conv_zoo` example
- minimal doc updates where needed

## Explicit Non-Goals

Do not include:

- heterogeneous operator work
- temporal operator work
- `edge_attr` support
- `MessagePassing` refactor
- full regression verification for the repository

## Repository Touchpoints

Phase 16 will mostly affect:

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

Phase 16 is complete when:

1. `EdgeConv`, `FeaStConv`, and `MFConv` are publicly exported
2. all three operators work on homogeneous graphs and support both call styles
3. the focused tests for exports, shapes, and hetero rejection pass
4. the compact integration test passes with the new operators
5. `conv_zoo.py` demonstrates the expanded operator set
6. no commits are made until explicit user approval

## Next Step

The next step is to write an implementation plan with exact file edits, focused test commands, and no-commit execution checkpoints.
