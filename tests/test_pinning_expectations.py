import torch

from tests.pinning import assert_tensor_pin_state


def test_assert_tensor_pin_state_accepts_driverless_fallback(monkeypatch):
    original = torch.tensor([1, 2, 3], dtype=torch.long)

    monkeypatch.setattr("tests.pinning.pin_memory_supported", lambda: False)

    assert_tensor_pin_state(original, original)


def test_assert_tensor_pin_state_accepts_pinned_tensor(monkeypatch):
    original = torch.tensor([1, 2, 3], dtype=torch.long)
    pinned = original.pin_memory()

    monkeypatch.setattr("tests.pinning.pin_memory_supported", lambda: True)

    assert_tensor_pin_state(pinned, original)
