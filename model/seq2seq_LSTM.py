import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from model.cnn_downsample import CNN_downsample

# Based on tutorial: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#training-the-model
# and https://www.youtube.com/watch?v=EoGUlvhRYpk&ab_channel=AladdinPersson

class Encoder(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            dropout_p: float,
            channels: list,
            stride: int,
            kernel_size: int,
            seq_len: int,
            downsample: bool,
            bidirection: bool = False
    ):
        """
        Parameters:
        -----------
        input_size: input tensor channel size
        hidden_size: LSTM hidden state channel size
        num_layers: number of LSTM layers
        dropout_p: LSTM dropout rate
        channels: list of number of channels corresponding to number of Conv1d layers
        stride: downsample Conv1d stride 
        kernel_size: downsample Conv1d kernel size
        seq_len: sequence length of data tensor
        downsample: bool to use CNN downsampling
        bidirection: bool for bi-directionality
        """
        super(Encoder, self).__init__()
        self.downsample = downsample
        if self.downsample:
            self.CNN_downsample = CNN_downsample(input_size, channels, stride, kernel_size, seq_len)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if isinstance(channels, int):
            channels = [channels]
        self.LSTM = nn.LSTM(channels[-1], hidden_size, num_layers, batch_first=True, dropout=dropout_p, bidirectional=bidirection)

    def forward(
            self,
            input, # input.shape() = (B, L, D)
    ):
        if self.downsample == True:
            # input, _ = torch.nn.utils.rnn.pad_packed_sequence(input, batch_first=True)
            input = self.CNN_downsample(input)
            # input = [input[i] for i in range(input.shape[0])]
            # input = torch.nn.utils.rnn.pack_sequence(input)
        output, (hidden, cell) = self.LSTM(input)
        
        return hidden, cell
    

class Decoder(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            num_layers: int,
            dropout_p: float,
            bidirection: bool
    ):
        """
        Parameters:
        -----------
        input_size: input tensor channel size
        hidden_size: LSTM hidden state channel size
        output_size: output tensor channel size
        num_layers: number of LSTM layers
        dropout_p: LSTM dropout rate
        bidirection: bool for bi-directionality
        """
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_p, bidirectional=bidirection)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(
            self,
            input, # shape (feature, hidden_size)
            hidden, # shape (num_layers, hidden_size)
            cell, # shape (num_layers, hidden_size)
    ):
        output, (hidden, cell) = self.LSTM(input, (hidden, cell))
        # output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # forward_output, backward_output = torch.split(output, split_size_or_sections=32, dim=2)

        pred = self.fc(output)
        pred = pred.squeeze(1) 
        # print(pred.shape)

        return pred, hidden, cell