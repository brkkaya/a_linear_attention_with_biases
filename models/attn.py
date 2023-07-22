import math
from typing import Optional
import torch.nn as nn
import torch


class Attention(nn.Module):
    def __init__(self, hid_dim: int, n_head: int, dropout: float = 0.1, bias: bool = False) -> None:
        super(Attention, self).__init__()
        self.hid_dim = hid_dim
        self.n_head = n_head
        self.dropout = dropout
        self.head_dim = hid_dim // n_head
        assert hid_dim % n_head == 0
        self.q = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.k = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.v = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.out = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):

        qk = torch.matmul(q, k.transpose(-2, -1))  # batch_size x n_head x seq_len x seq_len

        attn = qk / math.sqrt(self.head_dim)
        # adding alibi biases to qk, this does not break the mask
        if mask is not None:
            mask = self.prepare_mask(mask)
            attn = attn.masked_fill(mask == 0, -1e9)

        scaled_attn = torch.softmax(attn, dim=-1)
        scaled_attn = torch.matmul(scaled_attn, v)
        return scaled_attn

    def prepare_mask(self, mask: torch.Tensor):
        return mask[:, None, None, :]  # batch_size x seq_len

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = q.shape
        q = self.q(q)  # Linear transformation for q
        k = self.k(k)  # Linear transformation for k
        v = self.v(v)  # Linear transformation for v

        query, key, value = (
            q.view(batch_size, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3) for q in [q, k, v]
        )  # batch_size x seq_len x n_head x head_dim

        # # batch_size x seq_len x n_head x head_dim
        scaled_attn = self.scaled_dot_product(query, key, value, mask)  # apply scaled dot product
        scaled_attn = scaled_attn.permute(0, 2, 1, 3)
        attn = scaled_attn.reshape(-1, seq_len, self.hid_dim)

        attn = self.dropout(attn)  # apply dropout as paper says
        return self.out(attn)
