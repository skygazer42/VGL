# Homogeneous Convolution Expansion Phase 6 Design

**Date:** 2026-03-15
**Status:** Approved for planning

## Goal

Expand `vgl` with a first additional batch of real homogeneous GNN operators by adding `GINConv`, `GATv2Conv`, and `APPNPConv` as stable public layers that integrate cleanly with the existing graph and training abstractions.

## Scope Decisions

- Focus on homogeneous graph operators only
- Add three new operators in one coherent batch:
  - `GINConv`
  - `GATv2Conv`
  - `APPNPConv`
- Keep the public API consistent with existing `GCNConv`, `SAGEConv`, and `GATConv`
- Continue supporting both `conv(graph)` and `conv(x, edge_index)` call patterns
- Prioritize real usability over full PyG parameter-surface parity
- Keep message-passing internals pragmatic rather than refactoring the whole base class first

## Chosen Direction

Three directions were considered:

1. Add a broad mixed batch of unrelated layers
2. Add one focused homogeneous operator batch with coherent API conventions
3. Pause operator work and first redesign `MessagePassing` into a more general meta-framework

The chosen direction is:

> Add one focused homogeneous operator batch with coherent API conventions and only the minimal internal refactoring needed to make those layers correct.

This delivers visible algorithm breadth without destabilizing the framework core. A broad mixed batch would fragment the design, and a full `MessagePassing` rewrite would delay user-facing value.

## Architecture

Phase 6 should extend `vgl.nn.conv` with three new layers:

- `GINConv`
- `GATv2Conv`
- `APPNPConv`

These layers should live next to the existing homogeneous operators under:

- `vgl/nn/conv/gin.py`
- `vgl/nn/conv/gatv2.py`
- `vgl/nn/conv/appnp.py`

They should be exported from:

- `vgl.nn.conv`
- `vgl.nn`
- `vgl`

The package should not create a second algorithm namespace such as `advanced`, `experimental`, or `models`. The stable operator surface should remain simple and obvious.

## Public API Shape

The public import story should be:

```python
from vgl.nn.conv import GINConv, GATv2Conv, APPNPConv
```

and also:

```python
from vgl import GINConv, GATv2Conv, APPNPConv
```

The user-facing construction pattern should mirror the current operator style:

```python
conv = GINConv(in_channels=64, out_channels=64, eps=0.0, train_eps=False)
out = conv(graph)

conv = GATv2Conv(in_channels=64, out_channels=32, heads=4, concat=True)
out = conv(graph)

conv = APPNPConv(in_channels=64, out_channels=64, steps=10, alpha=0.1)
out = conv(graph)
```

Phase 6 should keep the operator surface intentionally smaller than PyG. The goal is stable usability, not full option parity.

## Operator Semantics

### GINConv

`GINConv` should expose:

- `in_channels`
- `out_channels`
- `eps=0.0`
- `train_eps=False`

The first implementation should use an internal MLP rather than accepting an arbitrary user-provided `nn` module. This keeps the public constructor stable and easy to document.

The core semantic should be:

- aggregate neighbor features by sum
- combine them with self features using the `1 + eps` weighting
- pass the result through the internal MLP

Output shape:

- `[num_nodes, out_channels]`

### GATv2Conv

`GATv2Conv` should expose:

- `in_channels`
- `out_channels`
- `heads=1`
- `concat=True`

Phase 6 should not include:

- dropout
- residual connections
- edge features
- bias toggles

The important stable behavior is the head output rule:

- if `concat=True`, output shape is `[num_nodes, out_channels * heads]`
- if `concat=False`, output shape is `[num_nodes, out_channels]`

### APPNPConv

`APPNPConv` should expose:

- `in_channels`
- `out_channels`
- `steps=10`
- `alpha=0.1`

In `vgl`, `APPNPConv` should be a layer with its own projection plus personalized propagation. That keeps it consistent with the rest of the operator surface instead of making it a special-case propagation utility.

The core semantic should be:

- project input node features to `out_channels`
- iteratively propagate for `steps`
- mix propagated state with the original projected state through `alpha`

Output shape:

- `[num_nodes, out_channels]`

## Message Passing Boundary

Phase 6 should remain pragmatic about the `MessagePassing` base class.

Rules:

- reuse `MessagePassing` where it keeps code simple and correct
- allow individual layers to implement custom `forward(...)` logic when the base class is not expressive enough
- do not redesign the whole message-passing abstraction just to force these three layers through one generic path

This matters especially for `GATv2Conv`, where attention logic is different enough that forcing it through the current base class would either distort the implementation or force an oversized refactor.

## Graph and Input Constraints

Phase 6 should enforce these constraints clearly:

- only homogeneous graphs are supported by these new layers
- all three layers should accept `conv(graph)` and `conv(x, edge_index)`
- heterogeneous graphs should fail early with clear errors
- output dtype and device should follow input tensors

These constraints preserve current API coherence and avoid silently incorrect behavior on typed graphs.

## Testing Strategy

Phase 6 tests should cover three layers:

### Operator contract tests

In `tests/nn/test_convs.py` add coverage for:

- `GINConv` on `Graph`
- `GINConv` on `(x, edge_index)`
- `GATv2Conv` head output shapes for `concat=True` and `concat=False`
- `APPNPConv` on `Graph`
- early failure on heterogeneous graph input for the new homogeneous operators

### Public export tests

`tests/test_package_exports.py` should assert that:

- `GINConv`
- `GATv2Conv`
- `APPNPConv`

are exposed from the root package.

### Integration and example tests

Add one integration path that proves each new operator can plug into the current homogeneous training loop rather than only passing isolated tensor-shape tests.

The integration should favor one compact training surface over three separate large integration files.

## Example Surface

Phase 6 should add one example dedicated to the new operator batch:

- `examples/homo/conv_zoo.py`

This example should:

- build one small homogeneous graph
- define tiny models using `GINConv`, `GATv2Conv`, and `APPNPConv`
- train each with the current `Trainer`
- print small result summaries

Existing examples should remain unchanged unless a small import/export adjustment is necessary.

## Phase 6 Deliverables

Phase 6 should ship:

- `GINConv`
- `GATv2Conv`
- `APPNPConv`
- exports from `vgl.nn.conv`, `vgl.nn`, and `vgl`
- shape and error-path tests for the new layers
- one integration path proving training-loop compatibility
- one algorithm-zoo style homogeneous example
- documentation updates

## Explicit Non-Goals

Do not include:

- heterogeneous operator redesign
- temporal operator redesign
- full PyG parameter parity
- edge-feature-aware attention
- dropout and residual option matrices
- a `MessagePassing` meta-refactor
- a large operator zoo beyond the chosen three layers

These belong to later phases and should not distort the first stable operator-expansion batch.

## Repository Touchpoints

Phase 6 will mostly affect:

- `vgl/nn/conv/`
- `vgl/nn/__init__.py`
- `vgl/__init__.py`
- `tests/nn/`
- `tests/integration/`
- `tests/test_package_exports.py`
- `examples/homo/`
- `docs/`

## Stability Constraints

Phase 6 should follow four rules:

1. Keep the operator API shape consistent with the current conv surface
2. Keep the new batch homogeneous-only
3. Keep internal implementation pragmatic rather than over-generalized
4. Keep public exports obvious and centralized

These constraints preserve room for future hetero and temporal operator work without forcing an early abstraction reset.

## Acceptance Criteria

Phase 6 is complete when:

1. `GINConv`, `GATv2Conv`, and `APPNPConv` are publicly exported
2. all three operators work on homogeneous graphs and support the expected call styles
3. `GATv2Conv` head output shapes are stable and tested
4. at least one integration path proves current training-loop compatibility
5. at least one example demonstrates the three new operators
6. existing tests and examples continue to pass

## Next Step

The next step is to create a detailed implementation plan with exact files, tests, commands, and commit checkpoints.
