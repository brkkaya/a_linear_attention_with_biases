import math
import torch.nn as nn
import torch
from einops import rearrange


class MHA_aLiBi(nn.Module):
    def __init__(self, dim_model: int, n_head: int, ctx_len: int) -> None:
        super(MHA_aLiBi, self).__init__()
        self.dim_model = dim_model
        self.n_head = n_head
        self.ctx_len = ctx_len
        assert dim_model % n_head == 0
        self.wq = nn.Linear(dim_model, dim_model)
        self.wk = nn.Linear(dim_model, dim_model)
        self.wv = nn.Linear(dim_model, dim_model)
        self.out = nn.Linear(dim_model, dim_model)

    def alibi_bias(self, ctx_len: int):
        bias = torch.arange(ctx_len).unsqueeze(1) - torch.arange(ctx_len)
        bias = -bias.masked_fill_(bias < 0, 0)
        return bias[None]

    def get_slope(self) -> torch.Tensor:
        """
        Calculation on slope is different acc. to paper. In the paper, author use only head size is power of 2.
        But in reality the head size may be different than that. If we do not divide the algorithm in two parts,
        slopes cannot be calculated correctly.
        Further info is in readMe
        """
        closest_power_two = 2 ** math.floor(math.log2(self.n_head))
        slopes = torch.pow(1 / 2.0, torch.arange(1, closest_power_two + 1))

        if closest_power_two < self.n_head:
            intra_slopes = 1 / math.sqrt(2.0)
            pair_slopes = torch.pow(intra_slopes, torch.arange(1, 1 + 2 * (self.n_head - closest_power_two), 2))
            slopes, _ = torch.sort(torch.cat([slopes, pair_slopes]), descending=True)

        return slopes[:, None, None]

    def scaled_dot_product(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        qk = torch.matmul(q, k.transpose(2, 3))

        alibi_attn = qk + (self.alibi_bias(ctx_len=self.ctx_len) * self.get_slope())

        if mask is not None:
            alibi_attn = alibi_attn.masked_fill_(mask == 0, value=-1e9)
        alibi_attn = nn.functional.softmax(alibi_attn, dim=-1)

        return torch.matmul(alibi_attn, v)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        wq = self.wq(q)
        wk = self.wk(k)
        wv = self.wv(v)
        wq = rearrange(wq, "b ctx_len (head dim) -> b head ctx_len dim", head=self.n_head)
        wk = rearrange(wk, "b ctx_len (head dim) -> b head ctx_len dim", head=self.n_head)
        wv = rearrange(wv, "b ctx_len (head dim) -> b head ctx_len dim", head=self.n_head)
        attn = self.scaled_dot_product(wq, wk, wv, mask)
        attn = rearrange(attn, "b head ctx_len dim -> b ctx_len (head dim)")
        return self.out(attn)


def decoder_mask(ctx_len: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(ctx_len, ctx_len)).T
    return mask.view(1, 1, ctx_len, ctx_len)
