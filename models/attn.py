import math
from typing import Optional
import torch.nn as nn
import torch
from einops import rearrange


class AliBiAttention(nn.Module):
    def __init__(self, hid_dim: int, n_head: int, dropout: float = 0.1, bias: bool = False) -> None:
        super(AliBiAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_head = n_head
        self.dropout = dropout
        self.head_dim = hid_dim // n_head
        assert hid_dim % n_head == 0
        self.q = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.k = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.v = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.out = nn.Linear(hid_dim, hid_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):

        qk = torch.bmm(q, k.transpose(2, 3))
        # we do not scale on AliBi bias
        if mask is not None:
            qk = qk.masked_fill_(mask, -1e9)

        scaled_attn = torch.softmax(qk, dim=-1)
        scaled_attn = torch.bmm(scaled_attn, v)
        return scaled_attn

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = q.shape
        query = self.q(q)  # batch_size x seq_len x hid_dim
        key = self.q(k)  # batch_size x seq_len x hid_dim
        value = self.q(v)  # batch_size x seq_len x hid_dim
        query, key, value = (
            qkv.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2) for qkv in [query, key, value]
        ) 
        # # batch_size x seq_len x n_head x head_dim
        scaled_attn = self.scaled_dot_product(query, key, value, mask) # apply scaled dot product 

        attn = scaled_attn.view(-1, seq_len, self.hid_dim)

        attn = self.dropout(attn) # apply dropout as paper says
        return self.out(attn)
