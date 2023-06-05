#%%
import math
import torch.nn as nn
import torch
from einops import rearrange
from aLiBi import MHA_aLiBi


class EncoderLayer(nn.Module):
    def __init__(self, dim_model: int, n_head: int) -> None:
        super(EncoderLayer, self).__init__()
        self.mha = MHA_aLiBi(dim_model=dim_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4), nn.LayerNorm(dim_model * 4), nn.Linear(dim_model * 4, dim_model)
        )

    def forward(self, x, mask=None):
        attn = self.mha(x, x, x, mask)
        norm = self.norm1(x + attn)
        ffn = self.ffn(norm)
        out = self.norm2(norm + ffn)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, dim_model: int, n_head: int, is_decoder_only: bool = True) -> None:
        super(DecoderLayer, self).__init__()
        self.is_decoder_only = is_decoder_only
        self.self_mha = MHA_aLiBi(dim_model=dim_model, n_head=n_head)
        if not is_decoder_only:
            self.cross_mha = MHA_aLiBi(dim_model=dim_model, n_head=n_head)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4), nn.LayerNorm(dim_model * 4), nn.Linear(dim_model * 4, dim_model)
        )
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)

    def forward(self, q, k, v, mask=None):
        self_mha = self.self_mha(v, v, v, mask)
        attn = self.norm1(v + self_mha)
        if not self.is_decoder_only:
            attn_cross = self.cross_mha(q, k, attn)
            attn = self.norm2(attn_cross + attn)
        ffn = self.ffn(attn)
        ffn_norm = self.norm3(attn + ffn)
        return ffn_norm


class Encoder(nn.Module):
    def __init__(self, dim_model: int, n_head: int) -> None:
        super(Encoder, self).__init__()
        self.enc = nn.ModuleList(EncoderLayer(dim_model=dim_model, n_head=n_head))

    def forward(self, x, mask=None):
        return self.enc(x, mask)


class Decoder(nn.Module):
    def __init__(self, dim_model: int, n_head: int) -> None:
        super(Decoder, self).__init__()
        self.dec = nn.ModuleList(DecoderLayer(dim_model=dim_model, n_head=n_head))

    def forward(self, q, k, v, mask):
        return self.dec(q, k, v, mask)


class Transformer(nn.Module):
    def __init__(self, dim_model: int, n_head: int, ctx_len: int,vocab_size: int, is_decoder_only: bool = True) -> None:
        super(Transformer, self).__init__()
        self.dim_model = dim_model
        self.n_head = n_head
        self.ctx_len = ctx_len
        self.is_decoder_only = is_decoder_only

        self.tok_embedding = nn.Embedding()
        if not is_decoder_only:
            self.enc = 
            pass
