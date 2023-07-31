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
    def __init__(self, dim_model: int, n_head: int, n_layer: int, vocab_dim: int) -> None:
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_dim, dim_model)
        # Create a list of decoder layers
        self.dec = nn.ModuleList([DecoderLayer(dim_model=dim_model, n_head=n_head) for _ in range(n_layer)])
        self.lm_head = nn.Linear()

    def forward(self, x: torch.Tensor, mask: torch.Tensor, targets: torch.Tensor):
        # Return the decoder
        for decoder_layer in self.dec:
            x = decoder_layer(x, mask)
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
