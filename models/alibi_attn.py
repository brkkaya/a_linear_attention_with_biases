import math
from typing import Optional
import torch.nn as nn
import torch

DEVICE = "cuda"
class AliBiAttention(nn.Module):
    def __init__(self, hid_dim: int, n_head: int, dropout: float = 0.1, bias: bool = False) -> None:
        super(AliBiAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_head = n_head
        self.dropout = dropout
        self.head_dim = hid_dim // n_head
        self.dk = math.sqrt(hid_dim)
        assert hid_dim % n_head == 0
        self.qkv = nn.Linear(hid_dim, hid_dim * 3, bias=bias)
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
        # Get the main heads size
        main_heads_size = 2 ** int(math.log2(self.n_head))
        # Get the m values for the main heads
        m_main = 2.0 ** (-8.0 / main_heads_size)
        # Get the m values for the other heads
        m = torch.pow(m_main, torch.arange(1, 1 + main_heads_size))

        # If the main heads size is less than the n_head, add the m values for the intra_heads
        if main_heads_size < self.n_head:
            intra_heads = 2.0 ** (-4.0 / main_heads_size)
            intra_heads = torch.pow(intra_heads, torch.arange(1, 1 + 2 * (self.n_head - main_heads_size), 2))
            m = torch.cat([m, intra_heads])
        # Return the m values
        return m[None, :, None, None].to(DEVICE)

    def casual_mask(self, mask: torch.Tensor):
        seq_len = mask.shape[-1]
        return torch.tril(torch.ones(1, 1, seq_len, seq_len)).to(DEVICE)

    @torch.no_grad()
    def alibi_biases(self, seq_len: int):
        """
        This function calculates the lower triangular matrix of the alibi_biases.
        """
        return torch.tril(-torch.arange(0, seq_len).unsqueeze(1) + torch.arange(0, seq_len))[None].to(DEVICE)

    def scaled_dot_product(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        This function calculates the scaled dot product of q and k with the alibi_biases.
        """
        seq_len = q.shape[2]
        qk = torch.matmul(q, k.transpose(2, 3))  # batch_size x n_head x seq_len x seq_len

        attn = qk + (self.get_slopes() * self.alibi_biases(seq_len=seq_len))

        # If mask is not None, apply the mask to the attention
        if mask is not None:
            mask = self.prepare_mask(mask)
            attn = attn.masked_fill(mask == 0, -1e9)

        # Calculate the scaled attention
        scaled_attn = torch.softmax(attn, dim=-1)
        scaled_attn = torch.matmul(scaled_attn, v)
        return scaled_attn

    def prepare_mask(self, mask: torch.Tensor):
        """
        This function prepares the mask for the scaled_dot_product.
        """
        return mask[:, None, None, :]  # batch_size x seq_len

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, kv_cache: list = None):
        # x is a 3D tensor of shape (batch_size, seq_len, hidden_dim)
        batch_size = x.shape[0]
        # split the 3D tensor into 3 separate tensors
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        if kv_cache is not None:
            old_k, old_v = kv_cache
            key = torch.cat([old_k, key], dim=1)
            value = torch.cat([old_v, value], dim=1)
        current_cache = [key, value]

        # reshape the 3D tensors into batch_size x seq_len x n_head x head_dim
        query, key, value = (
            qkv.reshape(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3) for qkv in [query, key, value]
        )

        # # batch_size x seq_len x n_head x head_dim
        scaled_attn = self.scaled_dot_product(query, key, value, mask)  # apply scaled dot product

        # reshape the scaled_attn tensor into batch_size x seq_len x n_head x head_dim
        scaled_attn = scaled_attn.permute(0, 2, 1, 3)
        # apply dropout as paper says
        attn = scaled_attn.reshape(batch_size, -1, self.hid_dim)

        # apply dropout
        attn = self.dropout(attn)  # apply dropout as paper says
        # return the output of the layer
        return self.out(attn), current_cache
