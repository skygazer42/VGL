# Homogeneous Transformer and Weisfeiler-Lehman Convolution Pack Design

**Date:** 2026-03-15
**Status:** Approved for planning

## Goal

Expand `vgl` with one more compact homogeneous operator batch by adding `TransformerConv` and `WLConvContinuous` as stable public layers that fit the current graph and training abstractions.

## Scope Decisions

- Focus on homogeneous graph operators only
- Add one compact batch:
  - `TransformerConv`
  - `WLConvContinuous`
- Keep the public API aligned with current `vgl` conv usage patterns
- Continue supporting both `conv(graph)` and `conv(x, edge_index)`
- Keep the first version independent from `edge_attr`
- Reuse the current training loop, export surface, and `conv_zoo` example

## Chosen Direction

Three directions were considered:

1. Add `TransformerConv` plus `WLConvContinuous` as a compact mixed batch
2. Add `TransformerConv`, `WLConvContinuous`, and a larger propagation layer such as `TWIRLSConv`
3. Delay attention-style expansion and continue only with lighter propagation operators

The chosen direction is:

> Add `TransformerConv` plus `WLConvContinuous` as a compact mixed batch.

This fills a meaningful homogeneous attention gap while still adding one low-coupling equal-width refinement operator. It also avoids widening the graph contract to edge attributes or introducing a much larger parameter surface in the same phase.

## Source Alignment

This phase should correspond to the core public semantics of the official operator families while remaining `vgl`-sized:

- `TransformerConv` should track the PyG layer shape with multi-head dot-product attention, optional root weighting, optional beta-gated skip mixing, and the usual `concat` behavior
- `WLConvContinuous` should track the PyG continuous Weisfeiler-Lehman refinement idea as an equal-width neighbor-and-self smoothing operator

This phase should deliberately not aim for full paper-level or framework-level parity.

## Architecture

Phase 18 should extend `vgl.nn.conv` with:

- `TransformerConv`
- `WLConvContinuous`

These layers should live under:

- `vgl/nn/conv/transformer.py`
- `vgl/nn/conv/wlconv.py`

They should be exported from:

- `vgl.nn.conv`
- `vgl.nn`
- `vgl`

The built-in operator surface should remain flat. No new `attention`, `experimental`, or `refinement` namespace should be introduced.

## Public API Shape

The user-facing import path should be:

```python
from vgl.nn.conv import TransformerConv, WLConvContinuous
```

and also:

```python
from vgl import TransformerConv, WLConvContinuous
```

The constructors should be:

```python
conv = TransformerConv(
    in_channels=64,
    out_channels=32,
    heads=4,
    concat=True,
    beta=False,
    dropout=0.0,
    bias=True,
    root_weight=True,
)
out = conv(graph)

conv = WLConvContinuous()
out = conv(graph)
```

## Operator Semantics

### Shared contract

Both operators should:

- support homogeneous graphs only
- preserve input dtype and device
- support `conv(graph)` and `conv(x, edge_index)`
- fail early on heterogeneous graphs with `"homogeneous"` in the error message

### TransformerConv

`TransformerConv` should expose:

- `in_channels`
- `out_channels`
- `heads=1`
- `concat=True`
- `beta=False`
- `dropout=0.0`
- `bias=True`
- `root_weight=True`

The first version should:

- compute queries from destination-node features
- compute keys and values from source-node features
- run destination-normalized attention per head
- support:
  - `concat=True` returning `[num_nodes, heads * out_channels]`
  - `concat=False` returning `[num_nodes, out_channels]`
- support an optional root branch when `root_weight=True`
- support optional beta-gated mixing between root and aggregated message paths when `beta=True`
- apply dropout to normalized attention weights

The first version should not include:

- `edge_attr`
- `edge_dim`
- bipartite tuple input
- `return_attention_weights`

### WLConvContinuous

`WLConvContinuous` should expose:

- no required constructor arguments

The first version should:

- preserve feature width
- compute one degree-normalized neighborhood summary
- combine the root feature and neighborhood summary as a continuous WL-style refinement
- remain parameter-free

Output shape:

- `[num_nodes, channels]`

The first version should not include:

- explicit `edge_weight`
- bipartite tuple input
- size hints

## Internal Helper Boundary

Phase 18 should continue using `vgl/nn/conv/_homo.py` for tiny shared helpers only.

Acceptable additions:

- reuse of `coerce_homo_inputs(...)`
- reuse of `edge_softmax(...)`
- reuse of `mean_propagate(...)`
- tiny attention-weight dropout handling inside `TransformerConv`

Unacceptable additions:

- `MessagePassing` redesign
- a generalized attention runtime
- an edge-attribute contract

## Testing Strategy

Per approved scope, Phase 18 should use a compact testing surface.

### Contract tests

In `tests/nn/test_convs.py` add:

- `TransformerConv` on `Graph`
- `TransformerConv` on `(x, edge_index)`
- `TransformerConv` concat output-shape coverage
- `TransformerConv` beta-path coverage
- `WLConvContinuous` on `Graph`
- `WLConvContinuous` on `(x, edge_index)`
- heterogeneous graph rejection tests for both operators

The first version should also include narrow validation tests:

- `TransformerConv` rejects invalid `heads`
- `TransformerConv` rejects invalid `dropout`

### Export tests

`tests/test_package_exports.py` should assert that:

- `TransformerConv`
- `WLConvContinuous`

are exposed from the root package.

### Compact integration path

Extend `tests/integration/test_homo_conv_zoo.py` so both operators also run through the current homogeneous training loop.

### Example surface

Extend `examples/homo/conv_zoo.py` to include:

- `transformerconv`
- `wlconv`

## Verification Boundary

Phase 18 should verify only the touched operator surface:

- focused operator and export tests
- compact training-loop integration test
- `conv_zoo.py` smoke run

This phase should not require:

- full `pytest -v`
- `ruff check`
- `mypy`
- full example smoke suite

## Deliverables

Phase 18 should ship:

- `TransformerConv`
- `WLConvContinuous`
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
- bipartite input tuples
- `return_attention_weights`
- `TWIRLSConv`
- `MessagePassing` refactor
- full regression verification for the repository

## Repository Touchpoints

Phase 18 will mostly affect:

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

Phase 18 is complete when:

1. `TransformerConv` and `WLConvContinuous` are publicly exported
2. both operators work on homogeneous graphs and support both call styles
3. `TransformerConv` respects the expected concat and non-concat output shapes
4. `WLConvContinuous` returns equal-width outputs
5. the focused tests for exports, shapes, rejection paths, and compact parameter validation pass
6. the compact integration test passes with the new operators
7. `conv_zoo.py` demonstrates the expanded operator set
8. no commits are made until explicit user approval

## Next Step

The next step is to write an implementation plan with exact file edits, focused test commands, and no-commit execution checkpoints.
