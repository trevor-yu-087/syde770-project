import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Based on tutorial: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#training-the-model
# and https://www.youtube.com/watch?v=EoGUlvhRYpk&ab_channel=AladdinPersson

class Encoder(nn.Module):
    def __init__(
            self,
            input_size: int, # number of features
            hidden_size: int, # number of number of output values (in hidden layer)
            num_layers: int,
            dropout_p: float,
    ):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_p)

    def forward(
            self,
            input, # input.shape() = (B, L, D)
    ):
        
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
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=dropout_p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(
            self,
            input, # shape (feature, hidden_size)
            hidden, # shape (num_layers, hidden_size)
            cell, # shape (num_layers, hidden_size)
    ):
        outputs, (hidden, cell) = self.LSTM(input, (hidden, cell))
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        pred = self.fc(outputs)
        pred = pred.squeeze(1) 
        # print(pred.shape)

        return pred, hidden, cell