from dataclasses import dataclass, field

import torch


def _transfer_data(data: dict, *, device=None, dtype=None, non_blocking: bool = False) -> dict:
    transferred = {}
    for key, value in data.items():
        if not isinstance(value, torch.Tensor):
            transferred[key] = value
            continue
        can_cast = dtype is not None and (value.is_floating_point() or value.is_complex())
        if can_cast:
            transferred[key] = value.to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            )
        else:
            transferred[key] = value.to(device=device, non_blocking=non_blocking)
    return transferred


def _pin_data(data: dict) -> dict:
    return {
        key: value.pin_memory() if isinstance(value, torch.Tensor) else value
        for key, value in data.items()
    }


@dataclass(slots=True)
class NodeStore:
    type_name: str
    data: dict[str, torch.Tensor] = field(default_factory=dict)

    def __getattr__(self, name: str) -> torch.Tensor:
        try:
            return self.data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to(self, device=None, dtype=None, non_blocking: bool = False):
        return NodeStore(
            type_name=self.type_name,
            data=_transfer_data(
                self.data, device=device, dtype=dtype, non_blocking=non_blocking
            ),
        )

    def pin_memory(self):
        return NodeStore(type_name=self.type_name, data=_pin_data(self.data))


@dataclass(slots=True)
class EdgeStore:
    type_name: tuple[str, str, str]
    data: dict[str, torch.Tensor] = field(default_factory=dict)

    def __getattr__(self, name: str) -> torch.Tensor:
        try:
            return self.data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to(self, device=None, dtype=None, non_blocking: bool = False):
        return EdgeStore(
            type_name=self.type_name,
            data=_transfer_data(
                self.data, device=device, dtype=dtype, non_blocking=non_blocking
            ),
        )

    def pin_memory(self):
        return EdgeStore(type_name=self.type_name, data=_pin_data(self.data))
