import math
from typing import Optional
import torch.nn as nn
import torch


class AliBiAttention(nn.Module):
    def __init__(self, hid_dim: int, n_head: int, dropout: float = 0.1, bias: bool = False) -> None:
        super(AliBiAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_head = n_head
        self.dropout = dropout
        self.head_dim = hid_dim // n_head
        self.dk = math.sqrt(hid_dim)
        assert hid_dim % n_head == 0
        self.q = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.k = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.v = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.out = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def get_slopes(self):
        """In the paper explained they trained only n_head when its only power of 2.
        In their github repository which is quite complex to find algorithm quickly,
        they add a small adjustment for other head sizes.
        Main algorithm is up to if part, but the author solution was adding m values to other heads also.
        To reach paper: (Press et al. 2021) https://arxiv.org/pdf/2108.12409.pdf
        Returns:
            torch.Tensor: slope values of aLiBi
        """
        main_heads_size = 2 ** int(math.log2(self.n_head))
        m_main = 2.0 ** (-8.0 / main_heads_size)
        m = torch.pow(m_main, torch.arange(1, 1 + main_heads_size))

        if main_heads_size < self.n_head:
            intra_heads = 2.0 ** (-4.0 / main_heads_size)
            intra_heads = torch.pow(intra_heads, torch.arange(1, 1 + 2 * (self.n_head - main_heads_size), 2))
            m = torch.cat([m, intra_heads])
        return m[None, :, None, None]

    @torch.no_grad()
    def alibi_biases(self, seq_len: int):
        return torch.tril(-torch.arange(0, seq_len).unsqueeze(1) + torch.arange(0, seq_len))[None]

    def scaled_dot_product(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        seq_len = q.shape[2]
        qk = torch.matmul(q, k.transpose(2, 3))  # batch_size x n_head x seq_len x seq_len

        attn = qk + (self.get_slopes() * self.alibi_biases(seq_len=seq_len))
        
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
        query = self.q(q)  # batch_size x seq_len x hid_dim
        key = self.q(k)  # batch_size x seq_len x hid_dim
        value = self.q(v)  # batch_size x seq_len x hid_dim
        query, key, value = (
            qkv.reshape(batch_size, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)
            for qkv in [query, key, value]
        )
        # # batch_size x seq_len x n_head x head_dim
        scaled_attn = self.scaled_dot_product(query, key, value, mask)  # apply scaled dot product

        scaled_attn = scaled_attn.permute(0, 2, 1, 3)
        attn = scaled_attn.reshape(-1, seq_len, self.hid_dim)

        attn = self.dropout(attn)  # apply dropout as paper says
        return self.out(attn)