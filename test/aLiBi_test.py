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


class TestAliBiAttention:
    def test_alibi_biases(self):
        seq_len = 3
        hid_dim = 4
        n_head = 2
        dropout = 0.0

        alibi_attn = AliBiAttention(hid_dim, n_head, dropout)
        seq = torch.arange(seq_len)
        alibi_biases_output = alibi_attn.alibi_biases(seq_len)
        
        desired_bias = []
        assert alibi_biases_output.shape == (1, seq_len, seq_len)

    def test_prepare_mask(self):
        batch_size = 2
        seq_len = 3
        hid_dim = 4
        n_head = 2
        dropout = 0.0

        alibi_attn = AliBiAttention(hid_dim, n_head, dropout)
        mask = torch.randn(batch_size, seq_len, seq_len)
        prepared_mask = alibi_attn.prepare_mask(mask)

        assert prepared_mask.shape == (batch_size, 1, 1, seq_len)
