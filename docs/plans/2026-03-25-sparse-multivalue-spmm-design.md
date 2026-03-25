# Sparse Multi-Value SPMM Design

## Context

`vgl.sparse` now supports `SparseTensor.values` shaped `(nnz, ...)`, payload-preserving conversions, multi-dimensional reductions, multi-head `sddmm(...)`, and multi-dimensional `edge_softmax(...)`. One deliberate limitation still remains in the sparse runtime: `spmm(...)` rejects any sparse tensor whose values have trailing payload dimensions.

That was the right constraint for the previous batch because the container model had just changed. Now it is the clearest remaining inconsistency inside `vgl.sparse` itself: the sparse representation can carry multi-value edge payloads, but the primary sparse-dense matmul primitive cannot consume them.

## Scope Choice

Three plausible next slices were considered:

1. Leave `spmm(...)` scalar-only and move on to new APIs.
2. Extend `spmm(...)` to broadcast over sparse payload dimensions while keeping the dense input rank-2.
3. Jump directly to generalized `gspmm(...)` with reducer/operator variants.

Option 2 is the right batch. It keeps the API surface stable, removes the most obvious internal mismatch in the sparse backend, and remains small enough to verify with focused sparse tests plus a full repository regression run.

## Recommended Design

Keep the current `spmm(sparse, dense)` signature and the existing dense input contract:

- `dense` stays rank-2 with shape `(num_cols, features)`
- sparse values may be `None`, `(nnz,)`, or `(nnz, *payload_dims)`

Output semantics:

- value-less sparse tensors behave exactly as before and return `(num_rows, features)`
- scalar edge values behave exactly as before and return `(num_rows, features)`
- multi-value sparse tensors return `(num_rows, *payload_dims, features)`

Operationally, `spmm(...)` should gather `dense[col]`, reshape it to align with trailing sparse payload dimensions, multiply edge-wise, then accumulate along the row axis.

## Shape Semantics

For sparse values shaped `(nnz, heads)` and dense shaped `(num_cols, features)`, the output should be `(num_rows, heads, features)`.

For sparse values shaped `(nnz, heads, channels)` and dense shaped `(num_cols, features)`, the output should be `(num_rows, heads, channels, features)`.

This keeps the sparse payload dimensions intact and appends dense feature channels at the end.

## Validation Rules

`spmm(...)` should still fail early when:

- `dense.ndim != 2`
- `dense.size(0)` does not match the sparse column dimension

No new public arguments are needed in this batch.

## Testing Strategy

Add focused coverage for:

- scalar `spmm(...)` behavior staying unchanged
- multi-head sparse values producing `(rows, heads, features)` output
- higher-rank sparse payloads producing `(rows, *payload_dims, features)` output
- empty sparse tensors with multi-value payloads returning correctly shaped zeros
- layout-agnostic behavior through a compressed sparse input

Then update docs so `vgl.sparse` no longer claims `spmm(...)` is scalar-only, and run the full suite.

## Non-Goals

This batch should not include:

- generalized `gspmm(...)` reducers/operators
- sparse-sparse matmul
- rank-3+ dense input support
- custom sparse kernels or distributed changes
