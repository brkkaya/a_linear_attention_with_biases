import torch
import sys
import pathlib
import math

sys.path.append(pathlib.Path(__file__).parent.parent)

from models.alibi_attn import AliBiAttention


def assert_shape(actual_shape, expected_shape):
    assert actual_shape == expected_shape, f"Expected shape {expected_shape}, but got {actual_shape}"


def test_forward_pass():
    # Test forward pass with random input
    hid_dim = 128
    n_head = 4
    dropout = 0.1
    batch_size = 8
    seq_len = 10

    attention = AliBiAttention(hid_dim, n_head, dropout)

    q = torch.randn(batch_size, seq_len, hid_dim)
    k = torch.randn(batch_size, seq_len, hid_dim)
    v = torch.randn(batch_size, seq_len, hid_dim)
    mask = torch.ones(batch_size, seq_len)
    mask[:, 5:] = 0
    output = attention(q, k, v, mask)

    # Check output shape
    assert_shape(output.shape, (batch_size, seq_len, hid_dim))


def test_masking():
    # Test forward pass with mask
    hid_dim = 128
    n_head = 4
    dropout = 0.1
    batch_size = 8
    seq_len = 10

    attention = AliBiAttention(hid_dim, n_head, dropout)

    q = torch.randn(batch_size, seq_len, hid_dim)
    k = torch.randn(batch_size, seq_len, hid_dim)
    v = torch.randn(batch_size, seq_len, hid_dim)

    # Create a mask to mask the last 3 positions
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[:, -3:] = False

    output = attention(q, k, v, mask=mask)

    # Check that the last 3 positions have been masked (output should be close to zero)
    assert torch.allclose(output[:, -3:, :], torch.zeros(batch_size, 3, hid_dim), atol=1e-6)


def test_gradients():
    # Test gradients are correctly computed during backpropagation
    hid_dim = 128
    n_head = 4
    dropout = 0.1
    batch_size = 8
    seq_len = 10

    attention = AliBiAttention(hid_dim, n_head, dropout)

    q = torch.randn(batch_size, seq_len, hid_dim, requires_grad=True)
    k = torch.randn(batch_size, seq_len, hid_dim, requires_grad=True)
    v = torch.randn(batch_size, seq_len, hid_dim, requires_grad=True)

    output = attention(q, k, v)

    # Dummy loss function (sum of all elements in the output)
    loss = output.sum()
    loss.backward()

    # Check gradients are not None
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


def run_tests():
    test_forward_pass()
    test_masking()
    test_gradients()


if __name__ == "__main__":
    run_tests()
