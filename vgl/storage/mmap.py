import json
from pathlib import Path
from typing import Union

import numpy as np
import torch

from vgl.storage.base import TensorSlice

PathLike = Union[str, Path]


def _metadata_path(path: PathLike) -> Path:
    path = Path(path)
    return path.with_name(f"{path.name}.meta.json")


class MmapTensorStore:
    def __init__(self, path: PathLike):
        self._path = Path(path)
        self._tensor: torch.Tensor | None = None
        self._metadata: dict | None = None

    @staticmethod
    def save(path: PathLike, tensor: torch.Tensor) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        array = tensor.detach().cpu().contiguous().numpy()
        mapped = np.memmap(destination, mode="w+", dtype=array.dtype, shape=array.shape)
        mapped[...] = array
        mapped.flush()
        _metadata_path(destination).write_text(
            json.dumps(
                {
                    "shape": list(array.shape),
                    "dtype": array.dtype.name,
                },
                sort_keys=True,
                indent=2,
            )
        )

    def _read_metadata(self) -> dict | None:
        if self._metadata is not None:
            return self._metadata
        metadata_path = _metadata_path(self._path)
        if not metadata_path.exists():
            return None
        self._metadata = json.loads(metadata_path.read_text())
        return self._metadata

    def _load(self) -> torch.Tensor:
        if self._tensor is not None:
            return self._tensor
        metadata = self._read_metadata()
        if metadata is None:
            self._tensor = torch.load(self._path, map_location="cpu", weights_only=True)
            return self._tensor

        array = np.memmap(
            self._path,
            mode="r+",
            dtype=np.dtype(metadata["dtype"]),
            shape=tuple(metadata["shape"]),
        )
        self._tensor = torch.from_numpy(array)
        return self._tensor

    @property
    def shape(self) -> tuple[int, ...]:
        metadata = self._read_metadata()
        if metadata is not None:
            return tuple(metadata["shape"])
        return tuple(self._load().shape)

    @property
    def dtype(self) -> torch.dtype:
        metadata = self._read_metadata()
        if metadata is not None:
            return torch.from_numpy(np.empty((), dtype=np.dtype(metadata["dtype"]))).dtype
        return self._load().dtype

    def fetch(self, index: torch.Tensor) -> TensorSlice:
        index = torch.as_tensor(index, dtype=torch.long)
        tensor = self._load()
        return TensorSlice(index=index, values=tensor[index])
