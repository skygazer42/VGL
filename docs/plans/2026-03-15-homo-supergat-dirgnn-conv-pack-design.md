# Homogeneous Supervised Attention and Directed Wrapper Convolution Pack Design

**Date:** 2026-03-15
**Status:** Approved for planning

## Goal

Expand `vgl` with one more compact homogeneous operator batch by adding `SuperGATConv` and `DirGNNConv` as stable public layers that fit the current graph and training abstractions.

## Scope Decisions

- Focus on homogeneous graph operators only
- Add one compact batch:
  - `SuperGATConv`
  - `DirGNNConv`
- Keep the public API aligned with current `vgl` conv usage patterns
- Continue supporting both `conv(graph)` and `conv(x, edge_index)`
- Keep the first version independent from `edge_attr`
- Reuse the current training loop, export surface, and `conv_zoo` example

## Chosen Direction

Three directions were considered:

1. Add `SuperGATConv` plus `DirGNNConv` as a compact attention-and-direction batch
2. Add only `SuperGATConv` and postpone directed wrappers
3. Begin supporting special-input layers such as `DNAConv`

The chosen direction is:

> Add `SuperGATConv` plus `DirGNNConv` as a compact attention-and-direction batch.

This fills one high-value attention gap and one high-value directed-graph gap without widening the graph contract to edge attributes or history-style feature tensors.

## Source Alignment

This phase should correspond to the core public semantics of the official operator families while remaining `vgl`-sized:

- `SuperGATConv` should track the PyG layer shape with multi-head attention, `concat` behavior, selectable `'MX'` or `'SD'` attention styles, and `get_attention_loss()`
- `DirGNNConv` should track the PyG wrapper shape with an underlying convolution, convex mixing of in- and out-edge aggregations, and an optional root transform

This phase should deliberately not aim for full framework-level parity.

## Architecture

Phase 19 should extend `vgl.nn.conv` with:

- `SuperGATConv`
- `DirGNNConv`

These layers should live under:

- `vgl/nn/conv/supergat.py`
- `vgl/nn/conv/dirgnn.py`

They should be exported from:

- `vgl.nn.conv`
- `vgl.nn`
- `vgl`

The built-in operator surface should remain flat. No new `directed`, `wrapper`, or `experimental` namespace should be introduced.

## Public API Shape

The user-facing import path should be:

```python
from vgl.nn.conv import SuperGATConv, DirGNNConv
```

and also:

```python
from vgl import SuperGATConv, DirGNNConv
```

The constructors should be:

```python
conv = SuperGATConv(
    in_channels=64,
    out_channels=32,
    heads=4,
    concat=True,
    negative_slope=0.2,
    dropout=0.0,
    add_self_loops=True,
    bias=True,
    attention_type="MX",
)
out = conv(graph)
loss = conv.get_attention_loss()

base = GraphConv(in_channels=64, out_channels=32)
conv = DirGNNConv(base, alpha=0.5, root_weight=True)
out = conv(graph)
```

## Operator Semantics

### Shared contract

Both operators should:

- support homogeneous graphs only
- preserve input dtype and device
- support `conv(graph)` and `conv(x, edge_index)`
- fail early on heterogeneous graphs with `"homogeneous"` in the error message

### SuperGATConv

`SuperGATConv` should expose:

- `in_channels`
- `out_channels`
- `heads=1`
- `concat=True`
- `negative_slope=0.2`
- `dropout=0.0`
- `add_self_loops=True`
- `bias=True`
- `attention_type="MX"`

The first version should:

- support exactly:
  - `attention_type="MX"`
  - `attention_type="SD"`
- compute one linear projection shared across heads
- compute destination-normalized attention per head
- support:
  - `concat=True` returning `[num_nodes, heads * out_channels]`
  - `concat=False` returning `[num_nodes, out_channels]`
- optionally add self-loops before attention
- cache a compact self-supervised attention objective
- expose `get_attention_loss()` as a scalar tensor after forward

The first version should not include:

- `neg_edge_index`
- `batch`
- `neg_sample_ratio`
- `edge_sample_ratio`
- `is_undirected`
- `return_attention_weights`

### DirGNNConv

`DirGNNConv` should expose:

- `conv`
- `alpha=0.5`
- `root_weight=True`

The first version should:

- deep-copy the provided base homogeneous convolution into:
  - one in-direction branch
  - one out-direction branch
- run the base operator on the original edge direction
- run the base operator on the flipped edge direction
- mix both outputs with `alpha`
- optionally add a learned root transform
- preserve the output width of the wrapped operator

The first version should not include:

- arbitrary wrappers around layers that require extra mandatory runtime tensors
- heterogeneous base operators
- tuple inputs

## Internal Helper Boundary

Phase 19 should continue using `vgl/nn/conv/_homo.py` for tiny shared helpers only.

Acceptable additions:

- reuse of `coerce_homo_inputs(...)`
- reuse of `edge_softmax(...)`
- reuse of existing flat homogeneous operator contracts

Unacceptable additions:

- `MessagePassing` redesign
- a generalized wrapper runtime for every possible layer family
- an edge-attribute contract

## Testing Strategy

Per approved scope, Phase 19 should use a compact testing surface.

### Contract tests

In `tests/nn/test_convs.py` add:

- `SuperGATConv` on `Graph`
- `SuperGATConv` on `(x, edge_index)`
- `SuperGATConv` concat output-shape coverage
- `SuperGATConv` attention-loss coverage
- `DirGNNConv` on `Graph`
- `DirGNNConv` on `(x, edge_index)`
- `DirGNNConv` alpha-mixing path coverage
- heterogeneous graph rejection tests for both operators

The first version should also include narrow validation tests:

- `SuperGATConv` rejects invalid `heads`
- `SuperGATConv` rejects invalid `dropout`
- `SuperGATConv` rejects invalid `attention_type`
- `DirGNNConv` rejects invalid `alpha`
- `DirGNNConv` rejects base operators with unsupported forward contracts

### Export tests

`tests/test_package_exports.py` should assert that:

- `SuperGATConv`
- `DirGNNConv`

are exposed from the root package.

### Compact integration path

Extend `tests/integration/test_homo_conv_zoo.py` so both operators also run through the current homogeneous training loop.

### Example surface

Extend `examples/homo/conv_zoo.py` to include:

- `supergat`
- `dirgnn`

## Verification Boundary

Phase 19 should verify only the touched operator surface:

- focused operator and export tests
- compact training-loop integration test
- `conv_zoo.py` smoke run

This phase should not require:

- full `pytest -v`
- `ruff check`
- `mypy`
- full example smoke suite

## Deliverables

Phase 19 should ship:

- `SuperGATConv`
- `DirGNNConv`
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
- tuple inputs
- `DNAConv`
- full self-supervised negative-sampling configuration
- `MessagePassing` refactor
- full regression verification for the repository

## Repository Touchpoints

Phase 19 will mostly affect:

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

Phase 19 is complete when:

1. `SuperGATConv` and `DirGNNConv` are publicly exported
2. both operators work on homogeneous graphs and support both call styles
3. `SuperGATConv` respects the expected concat and non-concat output shapes
4. `SuperGATConv.get_attention_loss()` returns a scalar tensor after forward
5. `DirGNNConv` successfully wraps supported homogeneous base operators
6. the focused tests for exports, shapes, rejection paths, and compact parameter validation pass
7. the compact integration test passes with the new operators
8. `conv_zoo.py` demonstrates the expanded operator set
9. no commits are made until explicit user approval

## Next Step

The next step is to write an implementation plan with exact file edits, focused test commands, and no-commit execution checkpoints.
