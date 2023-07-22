import torch.nn as nn
from models.alibi_attn import AliBiAttention


class DecoderLayer(nn.Module):
    def __init__(self, dim_model: int, n_head: int, bias: bool = True, dropout: float = 0.1) -> None:
        super(DecoderLayer, self).__init__()
        self.self_mha = AliBiAttention(dim_model=dim_model, n_head=n_head, bias=bias, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4), nn.LayerNorm(dim_model * 4), nn.Linear(dim_model * 4, dim_model)
        )
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)

    def forward(self, q, k, v, mask=None):
        self_mha = self.self_mha(v, v, v, mask)
        attn = self.norm1(v + self_mha)
        ffn = self.ffn(attn)
        ffn_norm = self.norm2(attn + ffn)
        return ffn_norm


class Decoder(nn.Module):
    def __init__(self, dim_model: int, n_head: int, n_layer: int, is_decoder_only: bool) -> None:
        super(Decoder, self).__init__()
        self.dec = nn.ModuleList(
            [DecoderLayer(dim_model=dim_model, n_head=n_head, is_decoder_only=is_decoder_only) for _ in range(n_layer)]
        )

    def forward(self, q, k, v, mask):
        return self.dec(q, k, v, mask)