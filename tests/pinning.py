from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def pin_memory_supported() -> bool:
    probe = torch.tensor([0], dtype=torch.long)
    try:
        pinned = probe.pin_memory()
    except RuntimeError:
        return False
    return bool(pinned.is_pinned())


def assert_tensor_pin_state(tensor: torch.Tensor, original: torch.Tensor | None = None) -> None:
    if pin_memory_supported():
        assert tensor.is_pinned()
        return

    assert not tensor.is_pinned()
    if original is not None:
        assert tensor is original
