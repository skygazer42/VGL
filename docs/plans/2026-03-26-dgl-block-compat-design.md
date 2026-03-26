# DGL Block Compatibility Design

## Context

VGL now has a first-class relation-local `Block` abstraction plus `to_block(...)` and sampler-produced block outputs for homogeneous node and link workloads. The DGL compatibility layer, however, still only handles `Graph` objects. That leaves one practical migration gap: users can round-trip graphs to DGL, but they cannot round-trip VGL message-flow blocks to DGL blocks.

That is a meaningful parity gap because `Block` is the main place where VGL deliberately meets DGL-style message-flow semantics. Without a compatibility bridge, block-aware code can only stay inside VGL even when the surrounding migration path claims DGL interoperability.

## Scope Choice

Three slices were considered:

1. Extend the existing graph adapters so `from_dgl(...)` and `to_dgl(...)` transparently accept both `Graph` and `Block`.
2. Add dedicated block helpers and `Block` convenience methods while leaving `Graph` adapter return types untouched.
3. Attempt full multi-relation DGL block import/export in the same batch.

Option 2 is the right slice. It preserves the current `Graph.from_dgl(...)` / `Graph.to_dgl(...)` contract, avoids mixed `Graph | Block` returns from the existing adapter entry points, and keeps this batch focused on VGL's existing single-relation `Block` abstraction.

## Recommended API

Add a narrow additive compatibility surface:

- `vgl.compat.dgl.block_to_dgl(block)`
- `vgl.compat.dgl.block_from_dgl(dgl_block)`
- `Block.to_dgl()`
- `Block.from_dgl(dgl_block)`

Semantics:

- VGL `Block` exports as a DGL block via `dgl.create_block(...)`
- block node and edge features are preserved on export and import
- `src_n_id`, `dst_n_id`, and `e_id` survive the round-trip through node and edge features
- same-type source/destination relations stay distinguishable through the existing VGL `src_store_type` / `dst_store_type` reconstruction logic
- import is limited to single-relation DGL blocks, which matches what one VGL `Block` can represent today

## Construction Strategy

Export should be direct, not graph-mediated. VGL `Block.graph` already stores the exact bipartite structure and feature tensors needed for DGL block construction. `block_to_dgl(...)` should therefore:

1. Read the one canonical relation from the `Block`.
2. Use the block's local `edge_index` as the source/destination side structure for `dgl.create_block(...)`.
3. Pass `num_src_nodes` and `num_dst_nodes` from the block's source/destination frontier sizes.
4. Copy source-node, destination-node, and edge features onto the DGL block.

Import should reconstruct the VGL `Block` directly from the DGL block's source-side and destination-side node frames plus edge frame. For same-type relations, reconstruction should reuse the existing VGL convention of internal store names like `node__src` and `node__dst`.

## Deferred Scope

This batch should explicitly not attempt:

- multi-relation DGL block import as one VGL object
- changing `Graph.from_dgl(...)` to sometimes return `Block`
- sampler changes
- PyG block compatibility

If external DGL blocks contain multiple canonical edge types, the import helper should fail clearly rather than inventing a wider VGL block abstraction.

## Testing Strategy

Add focused adapter regressions for:

- homogeneous VGL block export to a DGL block and import back to `Block`
- relation-local heterogeneous VGL block export to a DGL block and import back to `Block`
- import of an external single-relation DGL block with preserved features and IDs
- explicit failure on multi-relation DGL block import if that shape is represented in the fake adapter test harness

Then rerun the DGL adapter suite and the full repository suite before merge.
