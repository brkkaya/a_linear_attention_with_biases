import torch
import torch.nn as nn
from models.alibi_attn import AliBiAttention


class DecoderLayer(nn.Module):
    def __init__(self, dim_model: int, n_head: int, bias: bool = True, dropout: float = 0.1) -> None:
        """Transformers decoder layer"""
        super(DecoderLayer, self).__init__()
        # Create a self-attention module
        self.self_mha = AliBiAttention(dim_model=dim_model, n_head=n_head, bias=bias, dropout=dropout)

        # Create a feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4),
            nn.LayerNorm(dim_model * 4),
            nn.Linear(dim_model * 4, dim_model),
        )
        # Create a normalization layer
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # Calculate the self-attention
        self_mha = self.self_mha(x, mask)
        # Calculate the attention
        attn = self.norm1(x + self_mha)
        # Calculate the feed-forward network
        ffn = self.ffn(attn)
        # Calculate the feed-forward network norm
        ffn_norm = self.norm2(attn + ffn)
        # Return the feed-forward network norm
        return ffn_norm


class Decoder(nn.Module):
    def __init__(self, dim_model: int, n_head: int, n_layer: int, is_decoder_only: bool) -> None:
        super(Decoder, self).__init__()
        # Create a list of decoder layers
        self.dec = nn.ModuleList(
            [DecoderLayer(dim_model=dim_model, n_head=n_head, is_decoder_only=is_decoder_only) for _ in range(n_layer)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # Return the decoder
        return self.dec(x, mask)
