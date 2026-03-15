# Homogeneous Generalized and Stable Convolution Pack Design

**Date:** 2026-03-15
**Status:** Approved for planning

## Goal

Expand `vgl` with one more compact homogeneous operator batch by adding `PNAConv`, `GeneralConv`, and `AntiSymmetricConv` as stable public layers that fit the current graph and training abstractions.

## Scope Decisions

- Focus on homogeneous graph operators only
- Add one compact batch:
  - `PNAConv`
  - `GeneralConv`
  - `AntiSymmetricConv`
- Keep the public API aligned with current `vgl` conv usage patterns
- Continue supporting both `conv(graph)` and `conv(x, edge_index)`
- Keep the first version independent from `edge_attr`
- Reuse the current training loop, export surface, and `conv_zoo` example

## Chosen Direction

Three directions were considered:

1. Add only `PNAConv` and `GeneralConv`, and postpone stable deep propagation
2. Add `PNAConv`, `GeneralConv`, and `AntiSymmetricConv` with a first-version minimal parameter surface
3. Jump directly to a broader compatibility pass with `edge_attr`, `towers`, and more paper-level options

The chosen direction is:

> Add `PNAConv`, `GeneralConv`, and `AntiSymmetricConv` with a first-version minimal parameter surface.

This expands algorithm breadth in a way that still fits the current flat homogeneous API. It also avoids prematurely widening the graph data contract to edge attributes or introducing a larger helper/runtime abstraction.

## Source Alignment

This phase should correspond to the core public semantics of the official operator families while remaining `vgl`-sized:

- `PNAConv` should track the idea from PyG and DGL that multiple aggregators are combined with degree-based scalers, then projected
- `GeneralConv` should track the PyG design-space layer shape with configurable aggregation, optional directed messaging, optional attention, optional multi-head ensembles, and optional output normalization
- `AntiSymmetricConv` should track the PyG anti-symmetric update shape: residual Euler-style updates with an anti-symmetric linear term, damping, and a graph propagation branch

This phase should deliberately not aim for paper-level or framework-level parity.

## Architecture

Phase 17 should extend `vgl.nn.conv` with:

- `PNAConv`
- `GeneralConv`
- `AntiSymmetricConv`

These layers should live under:

- `vgl/nn/conv/pna.py`
- `vgl/nn/conv/generalconv.py`
- `vgl/nn/conv/antisymmetric.py`

They should be exported from:

- `vgl.nn.conv`
- `vgl.nn`
- `vgl`

The built-in operator surface should remain flat. No new `advanced`, `design_space`, `deep`, or `experimental` namespace should be introduced.

## Public API Shape

The user-facing import path should be:

```python
from vgl.nn.conv import PNAConv, GeneralConv, AntiSymmetricConv
```

and also:

```python
from vgl import PNAConv, GeneralConv, AntiSymmetricConv
```

The constructors should be:

```python
conv = PNAConv(
    in_channels=64,
    out_channels=32,
    aggregators=("sum", "mean", "max"),
    scalers=("identity", "amplification", "attenuation"),
    deg=None,
)
out = conv(graph)

conv = GeneralConv(
    in_channels=64,
    out_channels=32,
    aggr="add",
    skip_linear=False,
    directed_msg=True,
    heads=1,
    attention=False,
    l2_normalize=False,
)
out = conv(graph)

conv = AntiSymmetricConv(
    channels=64,
    num_iters=1,
    epsilon=0.1,
    gamma=0.1,
    act="tanh",
)
out = conv(graph)
```

## Operator Semantics

### Shared contract

All three operators should:

- support homogeneous graphs only
- preserve input dtype and device
- support `conv(graph)` and `conv(x, edge_index)`
- fail early on heterogeneous graphs with `"homogeneous"` in the error message

### PNAConv

`PNAConv` should expose:

- `in_channels`
- `out_channels`
- `aggregators=("sum", "mean", "max")`
- `scalers=("identity", "amplification", "attenuation")`
- `deg=None`

The first version should:

- support exactly the compact aggregator set:
  - `sum`
  - `mean`
  - `max`
- support exactly the compact scaler set:
  - `identity`
  - `amplification`
  - `attenuation`
- compute the requested neighborhood aggregations
- compute destination-node in-degree
- derive one scalar normalization reference from either:
  - `deg` if provided
  - the current graph if `deg is None`
- apply every requested scaler to every requested aggregation output
- concatenate all scaled aggregation outputs
- apply one projection-style update that also includes a root branch

Output shape:

- `[num_nodes, out_channels]`

The first version should not include:

- `edge_attr`
- `towers`
- `pre_layers` / `post_layers`
- larger aggregator families such as `min`, `var`, or `std`
- trainable normalization parameters

### GeneralConv

`GeneralConv` should expose:

- `in_channels`
- `out_channels`
- `aggr="add"`
- `skip_linear=False`
- `directed_msg=True`
- `heads=1`
- `attention=False`
- `l2_normalize=False`

The first version should:

- support exactly:
  - `aggr="add"`
  - `aggr="mean"`
  - `aggr="max"`
- support message construction from:
  - source features only if `directed_msg=False`
  - concatenated source and destination features if `directed_msg=True`
- support multi-head message ensembles by splitting output channels evenly across heads
- optionally score messages with destination-normalized attention when `attention=True`
- aggregate head outputs to destination nodes
- apply a skip/root branch
- optionally apply a learned linear skip when `skip_linear=True`
- optionally `l2`-normalize the output features

Output shape:

- `[num_nodes, out_channels]`

The first version should not include:

- `edge_attr`
- `attention_type`
- bipartite input tuples
- separate source/target channel sizes

### AntiSymmetricConv

`AntiSymmetricConv` should expose:

- `channels`
- `num_iters=1`
- `epsilon=0.1`
- `gamma=0.1`
- `act="tanh"`

The first version should:

- preserve feature width
- use one homogeneous graph propagation branch for `Phi(X, N_i)`
- learn one square weight matrix `W`
- form the anti-symmetric operator `W - W^T - gamma * I`
- apply the selected activation to the anti-symmetric branch plus propagation branch plus bias
- update node states with an Euler-style residual step for `num_iters` iterations

Output shape:

- `[num_nodes, channels]`

The first version should not include:

- custom injected `phi` modules
- edge weights
- broader activation resolver support beyond a compact string set

## Internal Helper Boundary

Phase 17 should continue using `vgl/nn/conv/_homo.py` for tiny shared helpers only.

Acceptable additions:

- a tiny node-degree helper
- a tiny degree-statistics helper for `PNAConv`
- reuse of `sum_propagate(...)`, `mean_propagate(...)`, `max_propagate(...)`, and `edge_softmax(...)`

Unacceptable additions:

- `MessagePassing` redesign
- a generalized operator runtime
- an edge-attribute contract

## Testing Strategy

Per approved scope, Phase 17 should use a compact testing surface.

### Contract tests

In `tests/nn/test_convs.py` add:

- `PNAConv` on `Graph`
- `PNAConv` on `(x, edge_index)`
- `GeneralConv` on `Graph`
- `GeneralConv` on `(x, edge_index)`
- `AntiSymmetricConv` on `Graph`
- `AntiSymmetricConv` on `(x, edge_index)`
- heterogeneous graph rejection tests for all three

The first version should also include a few narrow behavioral checks:

- `PNAConv` rejects unsupported aggregators and scalers
- `GeneralConv` rejects unsupported aggregators
- `GeneralConv` rejects invalid `heads` / `out_channels` combinations
- `AntiSymmetricConv` rejects unsupported activations

### Export tests

`tests/test_package_exports.py` should assert that:

- `PNAConv`
- `GeneralConv`
- `AntiSymmetricConv`

are exposed from the root package.

### Compact integration path

Extend `tests/integration/test_homo_conv_zoo.py` so all three operators also run through the current homogeneous training loop.

### Example surface

Extend `examples/homo/conv_zoo.py` to include:

- `pna`
- `generalconv`
- `antisymmetric`

## Verification Boundary

Phase 17 should verify only the touched operator surface:

- focused operator and export tests
- compact training-loop integration test
- `conv_zoo.py` smoke run

This phase should not require:

- full `pytest -v`
- `ruff check`
- `mypy`
- full example smoke suite

## Deliverables

Phase 17 should ship:

- `PNAConv`
- `GeneralConv`
- `AntiSymmetricConv`
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
- `PNAConv` tower decomposition
- `GeneralConv` bipartite input support
- custom `phi` injection for `AntiSymmetricConv`
- `MessagePassing` refactor
- full regression verification for the repository

## Repository Touchpoints

Phase 17 will mostly affect:

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

Phase 17 is complete when:

1. `PNAConv`, `GeneralConv`, and `AntiSymmetricConv` are publicly exported
2. all three operators work on homogeneous graphs and support both call styles
3. `PNAConv` and `GeneralConv` return `[num_nodes, out_channels]`
4. `AntiSymmetricConv` returns `[num_nodes, channels]`
5. the focused tests for exports, shapes, rejection paths, and compact parameter validation pass
6. the compact integration test passes with the new operators
7. `conv_zoo.py` demonstrates the expanded operator set
8. no commits are made until explicit user approval

## Next Step

The next step is to write an implementation plan with exact file edits, focused test commands, and no-commit execution checkpoints.
