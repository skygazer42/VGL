# Torch Sparse Interoperability Design

## Context

`vgl.sparse` already has its own `SparseTensor` abstraction with COO / CSR / CSC layouts, conversion helpers, and sparse ops such as `spmm(...)`, `sddmm(...)`, and `edge_softmax(...)`. That gives VGL an internal sparse backend, but it still has one practical gap at the boundary: there is no direct interoperability path with native PyTorch sparse tensors.

That matters because PyTorch is the execution substrate everywhere else in the project. Without a bridge, users who already hold `torch.sparse_coo_tensor`, `torch.sparse_csr_tensor`, or `torch.sparse_csc_tensor` objects have to unpack them manually, and VGL sparse results cannot be handed back into native sparse APIs cleanly.

## Scope Choice

Three slices were considered:

1. Full sparse-layout support including block compressed layouts such as BSR / BSC.
2. Rank-2 COO / CSR / CSC interop only.
3. Graph-level sparse export changes instead of low-level sparse-object interop.

Option 2 is the right slice.

It is the smallest additive bridge that matches VGL's existing sparse layouts exactly. Block-compressed layouts widen the surface beyond what `SparseTensor` can currently represent, and graph-level export changes can build on top of this lower-level sparse adapter later.

## Recommended API

Add:

- `vgl.sparse.from_torch_sparse(tensor)`
- `vgl.sparse.to_torch_sparse(sparse)`

Keep the scope intentionally narrow:

- tensors with exactly two sparse dimensions
- COO / CSR / CSC only
- preserve values and trailing payload dimensions
- preserve layout when possible

## Semantics

### Import

`from_torch_sparse(tensor)` should:

- accept `torch.sparse_coo`, `torch.sparse_csr`, and `torch.sparse_csc`
- reject dense tensors and unsupported sparse layouts
- require exactly two sparse dimensions, while allowing trailing dense payload dimensions
- import values exactly as stored on the input tensor
- preserve duplicate COO entries and current visible ordering

For COO tensors, that means the adapter should avoid coalescing implicitly, because coalescing can sum duplicate entries and reorder indices.

### Export

`to_torch_sparse(sparse)` should:

- return the native PyTorch sparse tensor matching the current VGL layout
- preserve trailing payload dimensions in `values`
- preserve device placement
- materialize unit values when `SparseTensor.values is None`, because native PyTorch sparse tensors always carry explicit values

That last point is the only intentional semantic widening in this batch. Inside VGL, structure-only sparse tensors behave like implicit ones in aggregate operations already, so explicit unit-value export stays consistent with observed behavior.

## Non-Goals

- block sparse layouts such as BSR / BSC
- higher-rank sparse tensors
- graph-level `Graph` convenience methods
- sparse autograd or kernel optimizations
- layout conversions beyond COO / CSR / CSC

## Testing Strategy

Add focused coverage for:

- importing external COO / CSR / CSC tensors into `SparseTensor`
- exporting VGL COO / CSR / CSC tensors back to native PyTorch sparse tensors
- preserving multi-dimensional sparse payloads
- structure-only export materializing unit values
- rejecting dense tensors and unsupported sparse layouts
- public sparse namespace exports

Then run focused sparse/package tests and the full suite.
