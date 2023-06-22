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

# Cross validation stuff
START_SUBJECT = 7
val_subjects = []
for i in range(START_SUBJECT, (START_SUBJECT+30), 5):
     val_subjects.append(i)

LSTM_OR_CNNLSTM = 0
params = {
    'hidden_size': [32, 64],
    'dropout': [0.137579431603837, 0.0725343342977065],
    'channels': [9, 64],
    'downsample': [False, True],
    'lr': [0.00239595953758425,0.0016079909971451242],
    'weight_decay': [0.00016730652977231463, 0.000100786933714903564],
    'epochs': [38, 35],
    'lstm_or_cnn-lstm': ['LSTM', 'CNN-LSTM'],
    'sample_period': [0.04, 0.02],
}

# Paths
SAVE_PATH = Path(f'outputs/cross-val/{params["lstm_or_cnn-lstm"][LSTM_OR_CNNLSTM]}/S{START_SUBJECT}-5-{val_subjects[-1]}/{datetime.now().strftime("%d-%m-%Y_%H%M%S")}')

TRAIN = True

if TRAIN == True:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=f'{SAVE_PATH}/tensorboard')
    TEST_PATH = SAVE_PATH
else:
    TEST_PATH = Path(input('Enter path to folder containing weights: '))

def main():
    # Get .csv files
    train_files, val_files, test_files = get_file_lists(
        val_sub_list=val_subjects,
        test_sub_list=[41],
        valid_files_path=Path(r'E:\smartwatch\subjects')
    )

    # Get dataloaders
    train_dataset = SmartwatchDataset(train_files, params['sample_period'][LSTM_OR_CNNLSTM])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hp.BATCH_SIZE, collate_fn=SmartwatchAugmentLstm(), drop_last=True, shuffle=True)

    val_dataset = SmartwatchDataset(val_files, params['sample_period'][LSTM_OR_CNNLSTM])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hp.BATCH_SIZE, collate_fn=SmartwatchAugmentLstm(), drop_last=True, shuffle=True)

    test_dataset = SmartwatchDataset(test_files, params['sample_period'][LSTM_OR_CNNLSTM])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hp.BATCH_SIZE, collate_fn=SmartwatchAugmentLstm(), drop_last=True, shuffle=False)

    # Initialize encoder and decoder
    encoder_model = Encoder(
        input_size=9,
        hidden_size=params['hidden_size'][LSTM_OR_CNNLSTM],
        num_layers=1,
        dropout_p=params['dropout'][LSTM_OR_CNNLSTM],
        channels=params['channels'][LSTM_OR_CNNLSTM],
        stride=2,
        kernel_size=63,
        seq_len=1024,
        downsample=params['downsample'][LSTM_OR_CNNLSTM],
    ).to(hp.DEVICE)
    decoder_model = Decoder(
        input_size=7,
        hidden_size=params['hidden_size'][LSTM_OR_CNNLSTM],
        output_size=7,
        num_layers=1,
        dropout_p=params['dropout'][LSTM_OR_CNNLSTM],
    ).to(hp.DEVICE)

    pytorch_total_params = sum(p.numel() for p in encoder_model.parameters())
    print(f"Encoder model params: {pytorch_total_params}")

    pytorch_total_params = sum(p.numel() for p in decoder_model.parameters())
    print(f"Decoder model params: {pytorch_total_params}")

    # Initialize loss functions
    loss_fn = nn.MSELoss()
    metric_loss_fn = nn.L1Loss()

    # Initialize optimizers
    encoder_optimizer = optim.Adam(encoder_model.parameters(), lr=params['lr'][LSTM_OR_CNNLSTM], weight_decay=params['weight_decay'][LSTM_OR_CNNLSTM])
    decoder_optimizer = optim.Adam(decoder_model.parameters(), lr=params['lr'][LSTM_OR_CNNLSTM], weight_decay=params['weight_decay'][LSTM_OR_CNNLSTM])
    if TRAIN == True:
        _ = LSTM_train_fn(
            train_loader,
            val_loader,
            encoder_model,
            decoder_model,
            encoder_optimizer,
            decoder_optimizer,
            loss_fn,
            metric_loss_fn,
            params['epochs'][LSTM_OR_CNNLSTM],
            hp.DEVICE,
            SAVE_PATH,
            writer,
            hp.TEACHER_FORCE_RATIO,
            checkpoint=None,
        )

    # test_LSTM(
    #     test_loader,
    #     encoder_model,
    #     decoder_model,
    #     loss_fn,
    #     metric_loss_fn,
    #     TEST_PATH,
    #     hp.DEVICE,
    # )

if __name__ == '__main__':
    main()