import json
import torch

from vgl.storage import MmapTensorStore


def test_mmap_tensor_store_round_trips_tensor_file(tmp_path):
    path = tmp_path / "tensor.pt"
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    MmapTensorStore.save(path, tensor)
    store = MmapTensorStore(path)
    result = store.fetch(torch.tensor([1, 2]))

    assert result.index.tolist() == [1, 2]
    assert torch.equal(result.values, torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
    assert store.shape == (3, 2)
    assert store.dtype is torch.float32


def test_mmap_tensor_store_writes_metadata_sidecar_and_fetches(tmp_path):
    path = tmp_path / "tensor.bin"
    metadata_path = tmp_path / "tensor.bin.meta.json"
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    MmapTensorStore.save(path, tensor)
    store = MmapTensorStore(path)
    result = store.fetch(torch.tensor([0, 2]))
    metadata = json.loads(metadata_path.read_text())

    assert path.exists()
    assert metadata_path.exists()
    assert metadata["shape"] == [3, 2]
    assert metadata["dtype"] == "float32"
    assert result.index.tolist() == [0, 2]
    assert torch.equal(result.values, torch.tensor([[1.0, 2.0], [5.0, 6.0]]))
    assert store.shape == (3, 2)
    assert store.dtype is torch.float32


def test_mmap_tensor_store_reads_legacy_torch_saved_tensor(tmp_path):
    path = tmp_path / "legacy.pt"
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    torch.save(tensor, path)
    store = MmapTensorStore(path)
    result = store.fetch(torch.tensor([1]))

    assert result.index.tolist() == [1]
    assert torch.equal(result.values, torch.tensor([[3.0, 4.0]]))
    assert store.shape == (3, 2)
    assert store.dtype is torch.float32
