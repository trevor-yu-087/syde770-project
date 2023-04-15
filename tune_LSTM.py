import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import os
import glob
from datetime import datetime
import optuna
from optuna import trial

import model.hyperparameters as hp
from model.seq2seq_LSTM import Encoder, Decoder
from utils.train import LSTM_train_fn
from utils.dataset import (
    SmartwatchDataset, 
    SmartwatchAugmentLstm, 
    get_file_lists
)
from utils.utils import test_LSTM
from torch.utils.tensorboard import SummaryWriter

# Get .csv files
train_files, val_files, test_files = get_file_lists(
    val_sub_list=['05', 10, 15, 20, 25, 30],
    test_sub_list=[35],
)

# Get dataloaders
train_dataset = SmartwatchDataset(train_files, sample_period=0.04)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hp.BATCH_SIZE, collate_fn=SmartwatchAugmentLstm(max_input_samples=512, downsample_output_seq=1), drop_last=True, shuffle=True)

val_dataset = SmartwatchDataset(val_files, sample_period=0.04)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hp.BATCH_SIZE, collate_fn=SmartwatchAugmentLstm(max_input_samples=512, downsample_output_seq=1), drop_last=True, shuffle=True)

test_dataset = SmartwatchDataset(test_files, sample_period=0.04)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hp.BATCH_SIZE, collate_fn=SmartwatchAugmentLstm(max_input_samples=512, downsample_output_seq=1), drop_last=True, shuffle=False)

def run(params=None,):
    SAVE_PATH = Path(f'outputs/{datetime.now().strftime("%d-%m-%Y_%H%M%S")}')
    writer = SummaryWriter(log_dir=f'{SAVE_PATH}/tensorboard')
    TEST_PATH = SAVE_PATH

    # Initialize encoder and decoder
    encoder_model = Encoder(
        input_size=9,
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout_p=params['dropout_p'],
        channels=params['channels'],
        stride=2,
        kernel_size=params['kernel_size'],
        seq_len=1024,
        downsample=False,
    ).to(hp.DEVICE)
    decoder_model = Decoder(
        input_size=7,
        hidden_size=params['hidden_size'],
        output_size=7,
        num_layers=params['num_layers'],
        dropout_p=params['dropout_p'],
    ).to(hp.DEVICE)

    # Initialize optimizers
    if params:
        print(params)
        encoder_optimizer = getattr(optim, params['optimizer'])(encoder_model.parameters(), lr= params['learning_rate'], weight_decay=params['weight_decay'])
        decoder_optimizer = getattr(optim, params['optimizer'])(decoder_model.parameters(), lr= params['learning_rate'], weight_decay=params['weight_decay'])
    else:
        encoder_optimizer = torch.optim.Adam(encoder_model.parameters(), lr=hp.ENCODER_LEARNING_RATE, weight_decay=0)
        decoder_optimizer = torch.optim.Adam(decoder_model.parameters(), lr=hp.DECODER_LEARNING_RATE, weight_decay=0)

    loss_fn = nn.MSELoss()
    metric_loss_fn = nn.L1Loss()

    val_loss_values = LSTM_train_fn(
            train_loader,
            val_loader,
            encoder_model,
            decoder_model,
            encoder_optimizer,
            decoder_optimizer,
            loss_fn,
            metric_loss_fn,
            params['num_epochs'],
            hp.DEVICE,
            SAVE_PATH,
            writer,
            hp.TEACHER_FORCE_RATIO,
            checkpoint=None,
        )
    
    return val_loss_values[-1]

def objective(trial):
    params = {
        'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128]),
        'num_layers': trial.suggest_int('num_layers', 1, 5),
        'dropout_p': trial.suggest_float('dropout_p', 0.05, 0.15),
        'channels': trial.suggest_categorical('channels', [[9]]),
        'kernel_size': trial.suggest_categorical('kernel_size', [7, 15, 31, 63]),
        'optimizer': trial.suggest_categorical('optimizer', ['Adam']),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-4, 1e-2),
        'num_epochs': trial.suggest_int('num_epoch', 30, 50),
    }

    accuracy = run(params)
    return accuracy

def main():
   study =  optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
   study.optimize(objective, n_trials=100)
   print("Number of finished trials: {}".format(len(study.trials)))

   best_trial = study.best_trial
   print(f"Best trial: {best_trial}")
   for key, value in best_trial.params.items():
      print("{}: {}".format(key, value))

if __name__=='__main__':
    main()