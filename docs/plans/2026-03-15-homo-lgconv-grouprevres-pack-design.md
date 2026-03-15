# Homogeneous LGConv and Grouped Reversible Residual Pack Design

**Date:** 2026-03-15
**Status:** Approved for planning

## Goal

Expand `vgl` with one more homogeneous operator batch by adding a PyG-aligned `LGConv` primitive and a lightweight grouped reversible residual wrapper `GroupRevRes` while preserving the current stable graph and training entrypoints.

## Scope Decisions

- Focus on homogeneous graph building blocks only
- Add one compact batch:
  - `LGConv`
  - `GroupRevRes`
- Keep the public API aligned with current `vgl` usage patterns
- Continue supporting both `layer(graph)` and `layer(x, edge_index)`
- Keep `edge_attr`, tuple inputs, and hetero support out of scope
- Keep `LightGCNConv` intact for backward compatibility
- Do not import PyG's custom invertible autograd machinery into `vgl`

## Chosen Direction

Three directions were considered:

1. Add `LGConv` and a lightweight reversible wrapper with PyG-style usage
2. Add only `LGConv` and postpone any reversible wrapper work
3. Rebuild the wrapper stack around a generic invertible runtime first

The chosen direction is:

> Add `LGConv` and a lightweight `GroupRevRes` now, but keep the implementation local and explicit instead of copying a heavy reversible runtime.

This keeps the user-facing experience close to PyG while still preserving `vgl`'s own controlled core abstractions and low internal churn.

## Source Alignment

This phase should correspond to the core public semantics of the official operator families while remaining `vgl`-sized:

- `LGConv` should track the PyG layer shape as a width-preserving propagation operator with optional symmetric normalization and no self-loops
- `GroupRevRes` should track the PyG grouped reversible residual usage pattern closely enough that users can seed it with one module plus `num_groups`, or provide an explicit `ModuleList`

This phase should deliberately not aim for full framework-level parity:

- no custom invertible autograd function
- no memory-saving storage tricks
- no arbitrary runtime argument chunking beyond the standard homogeneous `(x, edge_index)` contract

## Architecture

Phase 20 should extend `vgl` with:

- `vgl.nn.conv.LGConv`
- `vgl.nn.GroupRevRes`

These should live under:

- `vgl/nn/conv/lg.py`
- `vgl/nn/grouprevres.py`

They should be exported from:

- `vgl.nn.conv`
- `vgl.nn`
- `vgl`

`GroupRevRes` should not be placed under `vgl.nn.conv` because it is a wrapper block, not a message-passing operator family. The public surface should still stay flat at `vgl.nn` and `vgl`.

## Public API Shape

The user-facing import path should be:

```python
from vgl.nn.conv import LGConv
from vgl.nn import GroupRevRes
```

and also:

```python
from vgl import LGConv, GroupRevRes
```

The constructors should be:

```python
conv = LGConv(normalize=True)
out = conv(graph)

block = GroupRevRes(LGConv(), num_groups=2, split_dim=-1)
out = block(graph)
recovered = block.inverse(out, graph.edge_index)
```

The first version should also allow:

```python
from torch import nn

block = GroupRevRes(nn.ModuleList([LGConv(), LGConv()]), split_dim=-1)
```

## Operator Semantics

### Shared contract

Both additions should:

- support homogeneous graphs only
- preserve input dtype and device
- support both `layer(graph)` and `layer(x, edge_index)`
- fail early on heterogeneous graphs with `"homogeneous"` in the error message

### LGConv

`LGConv` should expose:

- `normalize=True`

The first version should:

- preserve feature width
- support one propagation step
- use symmetric degree normalization when `normalize=True`
- use plain additive propagation when `normalize=False`
- avoid adding self-loops

Output shape:

- `[num_nodes, channels]`

The first version should not include:

- edge weights
- sparse tensor branches
- self-loop options
- caching

`LGConv` is intentionally very close to the propagation primitive already represented by `LightGCNConv`, but it should exist as a first-class public name because PyG users expect it directly.

### GroupRevRes

`GroupRevRes` should expose:

- `conv`
- `num_groups=None`
- `split_dim=-1`

The first version should:

- accept either a seed module plus `num_groups`, or an explicit `nn.ModuleList`
- require at least `2` groups
- deep-copy the seed module when only one module is provided
- split the input feature tensor evenly across `split_dim`
- require the split dimension to be divisible by `num_groups`
- apply grouped additive reversible updates in forward
- expose `inverse(...)` to reconstruct the original tensor
- preserve total feature width

The grouped forward should follow the same high-level recurrence as PyG's grouped reversible block:

- initialize the running state from the sum of later groups
- update one group at a time with a wrapped operator call
- concatenate all group outputs back together along `split_dim`

The first version should not include:

- custom autograd memory savings
- arbitrary additional mandatory runtime tensors
- tuple feature inputs
- hetero graph support
- automatic support for wrapped modules that require arguments such as `x0`

## Internal Helper Boundary

Phase 20 should continue using `vgl/nn/conv/_homo.py` for tiny shared helpers only.

Acceptable additions:

- reusing `coerce_homo_inputs(...)`
- reusing `symmetric_propagate(...)`
- reusing `sum_propagate(...)`

Unacceptable additions:

- a `MessagePassing` redesign
- a generic invertible execution engine
- a new `models/` namespace just for one wrapper

## Testing Strategy

Per approved scope, Phase 20 should use a compact testing surface.

### Contract tests

In `tests/nn/test_convs.py` add:

- `LGConv` on `Graph`
- `LGConv` on `(x, edge_index)`
- `LGConv(normalize=False)` coverage
- hetero rejection coverage for `LGConv`

In `tests/nn/test_grouprevres.py` add:

- `GroupRevRes` on `Graph`
- `GroupRevRes` on `(x, edge_index)`
- inverse round-trip coverage
- validation coverage for invalid group count
- validation coverage for non-divisible channel width
- validation coverage for unsupported wrapped forward contracts
- hetero rejection coverage

### Export tests

`tests/test_package_exports.py` should assert that:

- `LGConv`
- `GroupRevRes`

are exposed from the root package.

### Compact integration path

Extend `tests/integration/test_homo_conv_zoo.py` so:

- `LGConv()`
- `GroupRevRes(LGConv(), num_groups=2)`

also run through the current homogeneous training loop.

### Example surface

Extend `examples/homo/conv_zoo.py` to include:

- `lgconv`
- `grouprevres`

## Verification Boundary

Phase 20 should verify only the touched operator surface:

- focused unit tests for `LGConv`, `GroupRevRes`, and exports
- compact training-loop integration test
- `conv_zoo.py` smoke run

This phase should not require:

- full `pytest -v`
- `ruff check`
- `mypy`
- repository-wide example smoke checks

## Deliverables

Phase 20 should ship:

- `LGConv`
- `GroupRevRes`
- exports from `vgl.nn.conv`, `vgl.nn`, and `vgl`
- focused contract tests
- compact integration coverage
- an expanded `conv_zoo` example
- minimal doc updates where needed

## Explicit Non-Goals

Do not include:

- heterogeneous wrapper work
- temporal wrapper work
- `edge_attr`
- custom invertible autograd storage tricks
- wrappers around operators with extra required runtime tensors
- a `MessagePassing` refactor
- commits before explicit user approval

## Repository Touchpoints

Phase 20 will mostly affect:

- `vgl/nn/conv/`
- `vgl/nn/grouprevres.py`
- `vgl/nn/__init__.py`
- `vgl/__init__.py`
- `tests/nn/`
- `tests/integration/`
- `tests/test_package_exports.py`
- `examples/homo/conv_zoo.py`
- `README.md`
- `docs/`

## Acceptance Criteria

Phase 20 is complete when:

1. `LGConv` and `GroupRevRes` are publicly exported
2. both additions support homogeneous graphs and both call styles
3. `LGConv(normalize=True)` provides normalized propagation and `normalize=False` provides raw additive propagation
4. `GroupRevRes.inverse(...)` reconstructs the original tensor for supported operators
5. focused tests for shapes, rejection paths, round-trip behavior, and exports pass
6. the compact integration test passes with `LGConv` and `GroupRevRes`
7. `conv_zoo.py` demonstrates the expanded surface
8. no commits are made until explicit user approval

## Next Step

The next step is to write an implementation plan with exact file edits, RED commands, and no-commit execution checkpoints.
