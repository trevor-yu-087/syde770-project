import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

# Based on tutorial: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#training-the-model
# and https://www.youtube.com/watch?v=EoGUlvhRYpk&ab_channel=AladdinPersson

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)

        pe[:, 0, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 0, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(torch.nn.Module):
    def __init__(self, input_size=9, output_size=7, d_model=32, n_heads=8, dropout=0.1, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        # INFO
        self.model_type = "Transformer"
        self.d_model = d_model

        self.input_transform = nn.Linear(input_size, d_model)
        self.target_transform = nn.Linear(output_size, d_model)

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            d_model=d_model, dropout=dropout, max_len=5000
        )

        
        #self.embedding = torch.nn.Embedding(num_tokens, d_model)
        

        self.transformer = torch.nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.out = torch.nn.Linear(d_model, output_size)

    def forward(
        self,
        src,
        tgt,
        src_padding=None,
        tgt_padding=None,
        tgt_lookahead=None
    ):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        src = self.input_transform(src)
        tgt = self.target_transform(tgt)

        # src = self.cnn(src)
        # src = self.layer_norm_1(src)
        # src = self.dropout(src)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        #src = self.embedding(src) * math.sqrt(self.dim_model)
        #tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # we permute to obtain size (sequence length, batch_size, dim_model),
        #src = src.permute(1, 0, 2)
        #tgt = tgt.permute(1, 0, 2)
        

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_lookahead, src_key_padding_mask=src_padding, tgt_key_padding_mask=tgt_padding)
        out = self.out(transformer_out)
        out = out.squeeze(1) 
        return out