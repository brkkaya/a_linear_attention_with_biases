import torch
import sys
import pathlib
import math

sys.path.append(pathlib.Path(__file__).parent.parent)

from models.alibi_attn import AliBiAttention


import torch
import torch.nn.functional as F
import torch.nn as nn
import math


def test_alibi_biases():
    seq_len = 5
    hid_dim = 4
    n_head = 2
    dropout = 0.0

    alibi_attn = AliBiAttention(hid_dim, n_head, dropout)
    alibi_biases_output = alibi_attn.alibi_biases(seq_len)

    desired_bias = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0],
            [-2, -1, 0, 0, 0],
            [-3, -2, -1, 0, 0],
            [-4, -3, -2, -1, 0],
        ],
        dtype=torch.int64,
    )[None]
    assert alibi_biases_output.shape == (1, seq_len, seq_len)
    assert alibi_biases_output.allclose(desired_bias)


def test_prepare_mask():
    batch_size = 2
    seq_len = 3
    hid_dim = 128
    n_head = 8
    dropout = 0.0

    alibi_attn = AliBiAttention(hid_dim, n_head, dropout)
    mask = torch.ones(batch_size, seq_len)
    prepared_mask = alibi_attn.prepare_mask(mask)

    assert prepared_mask.shape == (batch_size, 1, 1, seq_len)


if __name__ == "__main__":
    test_alibi_biases()
    test_prepare_mask()
