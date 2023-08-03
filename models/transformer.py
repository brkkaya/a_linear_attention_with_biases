import torch
import torch.nn as nn
from models.alibi_attn import AliBiAttention


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim: int, n_head: int, bias: bool = True, dropout: float = 0.1) -> None:
        """Transformers decoder layer"""
        super(DecoderLayer, self).__init__()
        # Create a self-attention module
        self.self_mha = AliBiAttention(hid_dim=hid_dim, n_head=n_head, bias=bias, dropout=dropout)

        # Create a feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 4),
            nn.LayerNorm(hid_dim * 4),
            nn.Linear(hid_dim * 4, hid_dim),
        )
        # Create a normalization layer
        self.norm1 = nn.LayerNorm(hid_dim)
        self.norm2 = nn.LayerNorm(hid_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, kv_cache: list = None):
        # Calculate the self-attention
        self_mha, kv_cache = self.self_mha(x, mask, kv_cache)
        # Calculate the attention
        attn = self.norm1(x + self_mha)
        # Calculate the feed-forward network
        ffn = self.ffn(attn)
        # Calculate the feed-forward network norm
        ffn_norm = self.norm2(attn + ffn)
        # Return the feed-forward network norm
        return ffn_norm, kv_cache


class Decoder(nn.Module):
    def __init__(self, hid_dim: int, n_head: int, n_layer: int, vocab_dim: int) -> None:
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_dim, hid_dim)
        # Create a list of decoder layers
        self.dec = nn.ModuleList([DecoderLayer(hid_dim=hid_dim, n_head=n_head) for _ in range(n_layer)])
        self.lm_head = nn.Linear(hid_dim, vocab_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, targets: torch.Tensor = None, kv_cache: list = None):
        # Return the decoder
        x = self.embedding(x)
        if not kv_cache:
            kv_cache = [None] * len(self.dec)
        new_kv_cache = []
        for decoder_layer, cache in zip(self.dec, kv_cache):
            # on training time, no harm to use a list of None values,
            # but in inference time, pre-calculated values are quite handy. To make inference quicker, we use it.
            x, updated_kv_cache = decoder_layer(x, mask, cache)
            new_kv_cache.append(updated_kv_cache)
            logits = self.lm_head(x)
        loss = None
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, new_kv_cache, loss

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool = True,
    ):
        # Generate tokens from the prompt
        generated = prompt
        kv_cache = None
        for _ in range(max_new_tokens):
            generation_size = generated.shape[-1]
            if kv_cache is None:
                mask = torch.ones(generated.shape)[:,generation_size:]
                logits, kv_cache, _ = self.forward(generated, mask=torch.ones(generated.shape), targets=None, kv_cache=kv_cache)
            else:
                logits, kv_cache, _ = self(generated[..., [-1]], mask=torch.ones(generated.shape), targets=None, kv_cache=kv_cache)

            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k, dim=-1, largest=True, sorted=True)
                logits[logits < v[:, [-1]]] = 0
            if top_p is not None:
                pass
                sorted_logits, _ = torch.sort(logits, descending=True)
                cumsum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                deletion_mask = cumsum_probs >= top_p
                sorted_logits[deletion_mask] = 0
            probs = torch.softmax(sorted_logits, dim=-1)
            if do_sample:
                token_next = torch.multinomial(probs, num_samples=1)
            else:
                token_next = probs[:, 0]
                generated = torch.cat([generated, token_next.unsqueeze(1)], dim=1)


# model = Decoder(48, 4, 2, 5200)
# prompt = torch.randint(0, 5200, size=(1,20))
# model.generate(prompt=prompt, max_new_tokens=20, do_sample=True, top_p=0.95, top_k=100, temperature=0.9)
