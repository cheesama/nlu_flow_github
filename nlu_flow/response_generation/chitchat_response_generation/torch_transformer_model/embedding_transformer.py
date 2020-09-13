from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm

import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        transformer_layers=2,
        num_encoder_layers=8,
        d_model=128,
        nhead=8,
        pad_token_id: int = 1,
    ):
        super(EmbeddingTransformer, self).__init__()
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

        self.encoder = nn.TransformerEncoder(
            TransformerEncoderLayer(d_model, nhead,),
            num_encoder_layers,
            LayerNorm(d_model),
        )

        self.transformer_layers = transformer_layers

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(self.max_seq_len, d_model)
        self.feature = nn.Linear(d_model, vocab_size)

        nn.init.xavier_uniform_(self.feature.weight)

    def forward(self, x, entity_labels=None):
        src_key_padding_mask = x == self.pad_token_id
        embedding = self.embedding(x)

        feature = embedding + self.position_embedding(torch.arange(x.size(1)).type_as(x)).repeat(x.size(0), 1, 1)

        for i in range(self.transformer_layers):
            # (N,S,E) -> (S,N,E) => (T,N,E) -> (N,T,E)
            #entity_feature = self.entity_encoder(entity_feature.transpose(1, 0)).transpose(1, 0)
            feature = self.encoder(feature.transpose(1, 0), src_key_padding_mask=src_key_padding_mask).transpose(1, 0)
            feature = feature.masked_fill(torch.isnan(feature), 0)

        pred = self.feature(feature)

        return pred
