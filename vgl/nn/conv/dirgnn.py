import copy
import inspect

from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs


def _infer_output_channels(conv):
    if hasattr(conv, "concat") and getattr(conv, "concat") and hasattr(conv, "heads") and hasattr(conv, "out_channels"):
        return conv.out_channels * conv.heads
    if hasattr(conv, "out_channels"):
        return conv.out_channels
    if hasattr(conv, "channels"):
        return conv.channels
    raise ValueError("DirGNNConv requires wrapped conv to expose out_channels or channels")


def _infer_input_channels(conv):
    candidates = [
        ("in_channels", None),
        ("linear", "in_features"),
        ("root_linear", "in_features"),
        ("neigh_linear", "in_features"),
        ("message_linear", "in_features"),
    ]
    for attr, child_attr in candidates:
        if not hasattr(conv, attr):
            continue
        value = getattr(conv, attr)
        if child_attr is None and isinstance(value, int):
            return value
        if child_attr is not None and hasattr(value, child_attr):
            return getattr(value, child_attr)
    raise ValueError("DirGNNConv could not infer wrapped conv input width")


def _validate_base_conv_signature(conv):
    parameters = list(inspect.signature(conv.forward).parameters.values())
    allowed_names = {"graph_or_x", "x", "edge_index"}
    if len(parameters) > 2:
        raise ValueError("DirGNNConv wrapped conv forward has unsupported extra runtime parameters")
    for parameter in parameters:
        if parameter.name not in allowed_names:
            raise ValueError("DirGNNConv wrapped conv forward has unsupported extra runtime parameters")


class DirGNNConv(nn.Module):
    def __init__(self, conv, alpha=0.5, root_weight=True):
        super().__init__()
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("DirGNNConv requires alpha to be in [0, 1]")
        _validate_base_conv_signature(conv)

        self.alpha = alpha
        self.root_weight = root_weight
        self.conv_in = copy.deepcopy(conv)
        self.conv_out = copy.deepcopy(conv)
        self.out_channels = _infer_output_channels(conv)

        for branch in (self.conv_in, self.conv_out):
            if hasattr(branch, "add_self_loops"):
                branch.add_self_loops = False
            if hasattr(branch, "root_weight") and isinstance(branch.root_weight, bool):
                branch.root_weight = False

        self.root_linear = None
        if root_weight:
            in_channels = _infer_input_channels(conv)
            self.root_linear = nn.Linear(in_channels, self.out_channels)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "DirGNNConv")
        forward_out = self.conv_in(x, edge_index)
        backward_out = self.conv_out(x, edge_index.flip(0))
        out = (self.alpha * forward_out) + ((1.0 - self.alpha) * backward_out)
        if self.root_linear is not None:
            out = out + self.root_linear(x)
        return out
