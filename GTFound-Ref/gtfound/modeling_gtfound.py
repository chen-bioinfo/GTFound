"""GTFound architecture for masked haplotype allele inference."""

from torch import nn

from .heads_gtfound import GTFoundLMHead
from .layers_gtfound import GTFoundLayer


class GTFoundModel(nn.Module):
    def __init__(
        self,
        vocab,
        embed_dim=768,
        num_attention_heads=12,
        num_layers=12,
        dropout=0.1,
    ):
        super().__init__()
        self.vocab = vocab
        self.embed_tokens = nn.Embedding(len(vocab), embed_dim)
        self.dropout_module = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                GTFoundLayer(
                    embedding_dim=embed_dim,
                    ffn_embedding_dim=4 * embed_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                    attention_dropout=0.1,
                    activation_dropout=0.1,
                )
                for _ in range(num_layers)
            ]
        )
        self.emb_layer_norm_before = nn.LayerNorm(embed_dim)
        self.emb_layer_norm_after = nn.LayerNorm(embed_dim)
        self.lm_head = GTFoundLMHead(
            embed_dim=embed_dim,
            output_dim=len(vocab),
            weight=self.embed_tokens.weight,
        )

    def forward(self, tokens, positions_embedding, ref, LD_data):
        """Return six token logits with shape [batch, sites, vocab_size]."""
        padding_mask = tokens.eq(self.vocab["[PAD]"])
        if not padding_mask.any():
            padding_mask = None

        x = self.embed_tokens(tokens) + positions_embedding
        x = self.dropout_module(self.emb_layer_norm_before(x))
        ref = self.embed_tokens(ref) + positions_embedding.unsqueeze(1)
        ref = self.dropout_module(self.emb_layer_norm_before(ref))
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        for layer in self.layers:
            x = layer(x, ref, LD_data, padding_mask=padding_mask)

        x = self.emb_layer_norm_after(x)
        return self.lm_head(x)
