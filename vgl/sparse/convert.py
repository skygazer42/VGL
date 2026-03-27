import torch

from vgl.sparse.base import SparseLayout, SparseTensor


def _sparse_shape_from_torch(tensor: torch.Tensor) -> tuple[int, int]:
    if not hasattr(tensor, "sparse_dim") or int(tensor.sparse_dim()) != 2:
        raise ValueError("torch sparse tensor must have exactly two sparse dimensions")
    return int(tensor.shape[0]), int(tensor.shape[1])


def _coo_indices_and_values(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Preserve duplicate entries and current ordering for uncoalesced COO inputs.
    if bool(tensor.is_coalesced()):
        return tensor.indices(), tensor.values()
    return tensor._indices(), tensor._values()


def _explicit_values_for_export(sparse: SparseTensor) -> torch.Tensor:
    if sparse.values is not None:
        return sparse.values
    device = None
    if sparse.layout is SparseLayout.COO:
        device = sparse.row.device
    elif sparse.layout is SparseLayout.CSR:
        device = sparse.crow_indices.device
    else:
        device = sparse.ccol_indices.device
    return torch.ones(sparse.nnz, dtype=torch.float32, device=device)


def from_torch_sparse(tensor: torch.Tensor) -> SparseTensor:
    layout = tensor.layout
    if layout is torch.sparse_coo:
        shape = _sparse_shape_from_torch(tensor)
        indices, values = _coo_indices_and_values(tensor)
        return SparseTensor(
            layout=SparseLayout.COO,
            shape=shape,
            row=indices[0],
            col=indices[1],
            values=values,
        )
    if layout is torch.sparse_csr:
        shape = _sparse_shape_from_torch(tensor)
        return SparseTensor(
            layout=SparseLayout.CSR,
            shape=shape,
            crow_indices=tensor.crow_indices(),
            col_indices=tensor.col_indices(),
            values=tensor.values(),
        )
    if layout is torch.sparse_csc:
        shape = _sparse_shape_from_torch(tensor)
        return SparseTensor(
            layout=SparseLayout.CSC,
            shape=shape,
            ccol_indices=tensor.ccol_indices(),
            row_indices=tensor.row_indices(),
            values=tensor.values(),
        )
    raise ValueError("from_torch_sparse expects a torch sparse tensor in COO, CSR, or CSC layout")


def from_edge_index(
    edge_index: torch.Tensor,
    *,
    shape: tuple[int, int],
    layout: SparseLayout = SparseLayout.COO,
    values: torch.Tensor | None = None,
) -> SparseTensor:
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    row = edge_index[0].to(dtype=torch.long)
    col = edge_index[1].to(dtype=torch.long)
    coo = SparseTensor(layout=SparseLayout.COO, shape=shape, row=row, col=col, values=values)
    if layout is SparseLayout.COO:
        return coo
    if layout is SparseLayout.CSR:
        return to_csr(coo)
    if layout is SparseLayout.CSC:
        return to_csc(coo)
    raise ValueError(f"Unsupported sparse layout: {layout}")


def to_coo(sparse: SparseTensor) -> SparseTensor:
    if sparse.layout is SparseLayout.COO:
        return sparse
    if sparse.layout is SparseLayout.CSR:
        counts = sparse.crow_indices[1:] - sparse.crow_indices[:-1]
        row = torch.repeat_interleave(
            torch.arange(sparse.shape[0], dtype=torch.long, device=sparse.crow_indices.device),
            counts,
        )
        return SparseTensor(
            layout=SparseLayout.COO,
            shape=sparse.shape,
            row=row,
            col=sparse.col_indices,
            values=sparse.values,
        )
    counts = sparse.ccol_indices[1:] - sparse.ccol_indices[:-1]
    col = torch.repeat_interleave(
        torch.arange(sparse.shape[1], dtype=torch.long, device=sparse.ccol_indices.device),
        counts,
    )
    return SparseTensor(
        layout=SparseLayout.COO,
        shape=sparse.shape,
        row=sparse.row_indices,
        col=col,
        values=sparse.values,
    )


def to_csr(sparse: SparseTensor) -> SparseTensor:
    if sparse.layout is SparseLayout.CSR:
        return sparse
    coo = to_coo(sparse)
    row = coo.row
    col = coo.col
    sort_key = row * max(coo.shape[1], 1) + col
    order = torch.argsort(sort_key)
    row = row[order]
    col = col[order]
    values = None if coo.values is None else coo.values[order]
    counts = torch.bincount(row, minlength=coo.shape[0])
    crow_indices = torch.zeros(coo.shape[0] + 1, dtype=torch.long, device=row.device)
    crow_indices[1:] = torch.cumsum(counts, dim=0)
    return SparseTensor(
        layout=SparseLayout.CSR,
        shape=coo.shape,
        crow_indices=crow_indices,
        col_indices=col,
        values=values,
    )


def to_csc(sparse: SparseTensor) -> SparseTensor:
    if sparse.layout is SparseLayout.CSC:
        return sparse
    coo = to_coo(sparse)
    row = coo.row
    col = coo.col
    sort_key = col * max(coo.shape[0], 1) + row
    order = torch.argsort(sort_key)
    row = row[order]
    col = col[order]
    values = None if coo.values is None else coo.values[order]
    counts = torch.bincount(col, minlength=coo.shape[1])
    ccol_indices = torch.zeros(coo.shape[1] + 1, dtype=torch.long, device=col.device)
    ccol_indices[1:] = torch.cumsum(counts, dim=0)
    return SparseTensor(
        layout=SparseLayout.CSC,
        shape=coo.shape,
        ccol_indices=ccol_indices,
        row_indices=row,
        values=values,
    )


def to_torch_sparse(sparse: SparseTensor) -> torch.Tensor:
    values = _explicit_values_for_export(sparse)
    size = tuple(sparse.shape) + tuple(values.shape[1:])
    if sparse.layout is SparseLayout.COO:
        indices = torch.stack((sparse.row, sparse.col))
        return torch.sparse_coo_tensor(indices, values, size=size)
    if sparse.layout is SparseLayout.CSR:
        return torch.sparse_csr_tensor(
            sparse.crow_indices,
            sparse.col_indices,
            values,
            size=size,
        )
    if sparse.layout is SparseLayout.CSC:
        return torch.sparse_csc_tensor(
            sparse.ccol_indices,
            sparse.row_indices,
            values,
            size=size,
        )
    raise ValueError(f"Unsupported sparse layout: {sparse.layout}")
