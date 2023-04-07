import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import os
import glob
from datetime import datetime

import model.hyperparameters as hp
from model.seq2seq_LSTM import Encoder, Decoder
from utils.train import LSTM_train_fn
from utils.dataset import (
    SmartwatchDataset, 
    SmartwatchAugmentLstm, 
    get_file_lists
)
from utils.utils import test_LSTM

# Paths
SAVE_PATH = Path(f'outputs/{datetime.now().strftime("%d-%m-%Y_%H%M%S")}')

TRAIN = False

if TRAIN == True:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=f'{SAVE_PATH}/tensorboard')
    TEST_PATH = SAVE_PATH
else:
    TEST_PATH = Path(input('Enter path to folder containing weights: '))

def main():
    # Get .csv files
    train_files, val_files, test_files = get_file_lists(
        val_sub_list=['05', 10, 15, 20, 25, 30],
        test_sub_list=[35],
    )

    # Get dataloaders
    train_dataset = SmartwatchDataset(train_files)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hp.BATCH_SIZE, collate_fn=SmartwatchAugmentLstm(), drop_last=True, shuffle=True)

    val_dataset = SmartwatchDataset(val_files)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hp.BATCH_SIZE, collate_fn=SmartwatchAugmentLstm(), drop_last=True, shuffle=True)

    test_dataset = SmartwatchDataset(test_files)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hp.BATCH_SIZE, collate_fn=SmartwatchAugmentLstm(), drop_last=True, shuffle=False)

    # Initialize encoder and decoder
    encoder_model = Encoder(
        input_size=9,
        hidden_size=32,
        num_layers=1,
        dropout_p=0.1,
    ).to(hp.DEVICE)
    decoder_model = Decoder(
        input_size=7,
        hidden_size=32,
        output_size=7,
        num_layers=1,
        dropout_p=0.1,
    ).to(hp.DEVICE)

    # Initialize loss functions
    loss_fn = nn.MSELoss()
    metric_loss_fn = nn.L1Loss()

    # Initialize optimizers
    encoder_optimizer = optim.Adam(encoder_model.parameters(), lr=hp.ENCODER_LEARNING_RATE)
    decoder_optimizer = optim.Adam(decoder_model.parameters(), lr=hp.DECODER_LEARNING_RATE)
    if TRAIN == True:
        LSTM_train_fn(
            train_loader,
            val_loader,
            encoder_model,
            decoder_model,
            encoder_optimizer,
            decoder_optimizer,
            loss_fn,
            metric_loss_fn,
            hp.NUM_EPOCH,
            hp.DEVICE,
            SAVE_PATH,
            writer,
            hp.TEACHER_FORCE_RATIO,
            checkpoint=None
        )

    test_LSTM(
        test_loader,
        encoder_model,
        decoder_model,
        loss_fn,
        metric_loss_fn,
        TEST_PATH,
        hp.DEVICE,
    )

if __name__ == '__main__':
    main()