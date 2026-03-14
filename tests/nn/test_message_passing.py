import torch

from gnn.nn.message_passing import MessagePassing


class IdentityMessagePassing(MessagePassing):
    pass


def test_message_passing_aggregates_neighbor_messages():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]])

    out = IdentityMessagePassing()(x, edge_index)

    assert out.shape == x.shape
    assert torch.equal(out, x.flip(0))
