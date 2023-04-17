import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

from model.cnn_downsample import CNN_downsample

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
    def __init__(
            self,
            input_size: int,
            dropout: float,
            stride: int,
            kernel_size: int,
            seq_len: int,
            downsample: bool,
            output_size=7, 
            d_model=32, 
            n_heads=8,
            num_encoder_layers=6, 
            num_decoder_layers=6
    ):
        super().__init__()
        # INFO
        self.model_type = "Transformer"
        self.d_model = d_model
        self.seq_len = seq_len
        self.stride = stride
        self.n_heads = n_heads

        self.downsample = downsample

        if self.downsample:
            self.CNN_downsample = CNN_downsample(input_size, d_model, stride, kernel_size, seq_len)
            #self.tgt_CNN_downsample = CNN_downsample(output_size, channels, stride, kernel_size, seq_len)
        else:
            self.input_transform = nn.Linear(input_size, d_model)
        self.target_transform = nn.Linear(output_size, d_model)

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            d_model=d_model, dropout=dropout, max_len=5000
        )

        self.transformer = torch.nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            dim_feedforward=d_model*2,
            activation="gelu"
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
        #src_lookahead_mask = enc_lookahead_mask.repeat(self.num_heads, 1, 1)
        if tgt_lookahead is not None:
            tgt_lookahead = tgt_lookahead.repeat(self.n_heads, 1, 1)


        if self.downsample:
            src = self.CNN_downsample(src)
            #tgt = self.tgt_CNN_downsample(tgt)
   
            if src_padding is not None:
                src_padding = src_padding.float().unsqueeze(1)
                #tgt_padding = tgt_padding.unsqueeze(1)
                #tgt_lookahead= tgt_lookahead.unsqueeze(1)

                src_padding = torch.nn.functional.interpolate(src_padding, size=[math.ceil(self.seq_len/self.stride)])[:,0,:].bool()

                #tgt_padding = torch.nn.functional.interpolate(tgt_padding.float(), size=[math.ceil(self.seq_len/self.stride)])[:,0,:].bool()
                #tgt_lookahead = torch.nn.functional.interpolate(tgt_lookahead, size=[math.ceil(self.seq_len/self.stride), math.ceil(self.seq_len/self.stride)])[:,0,:,:]

        else:
            src = self.input_transform(src)
        tgt = self.target_transform(tgt)

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