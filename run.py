import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import os
import json
import glob
from datetime import datetime
import typer

import model.hyperparameters as hp

from utils.train import LSTM_train_fn
from utils.dataset import (
    SmartwatchDataset, 
    SmartwatchAugmentLstm, 
    get_file_lists
)
from utils.utils import (
    run_lstm, 
    run_cnnlstm, 
    run_transformer, 
    run_cnntransformer, 
    test_LSTM,
    test_transformer,
)

app = typer.Typer()


@app.command()
def run(
    data_json: Path,
    save_dir: Path,
    model: str, # lstm, cnn-lstm, transformer, or cnn-transformer
    test_model: bool = False, # run test after training
    enable_checkpoints: bool = False,
):

    # create output save path
    SAVE_PATH = Path(f'{save_dir}/outputs/{model}/{datetime.now().strftime("%d-%m-%Y_%H%M%S")}')
    print(f'CUDA available: {torch.cuda.is_available()}')

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=f'{SAVE_PATH}/tensorboard')
    TEST_PATH = SAVE_PATH

    # parameters for models
    if model == 'lstm' or model =='transformer':
        sample_period = 0.04
        downsample = False
    elif model == 'cnn-lstm' or model == 'cnn-transformer':
        sample_period = 0.02
        downsample = True
    else:
        raise Exception('Unsupported model type')

    # get dataloaders
    with data_json.open('r') as f:
        valid_files = json.loads(f.read())
    f.close()
    train_files = valid_files['train']
    val_files = valid_files['val']
    test_files = valid_files['test']

    train_dataset = SmartwatchDataset(train_files, sample_period)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hp.BATCH_SIZE, collate_fn=SmartwatchAugmentLstm(), drop_last=True, shuffle=False)

    val_dataset = SmartwatchDataset(val_files, sample_period)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hp.BATCH_SIZE, collate_fn=SmartwatchAugmentLstm(), drop_last=True, shuffle=False)

    test_dataset = SmartwatchDataset(test_files, sample_period)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hp.BATCH_SIZE, collate_fn=SmartwatchAugmentLstm(), drop_last=True, shuffle=False)
    
    lstm_params = {
        'hidden_size': [32, 64],
        'dropout': [0.137579431603837, 0.1],
        'channels': [9, 64],
        'lr': [0.00239595953758425, 1e-3],
        'weight_decay': [0.00016730652977231463, 0.000100786933714903564],
        'epochs': [38, 35],
    }

    transformer_params = {
    'hidden_size': [32, 128],
    'dropout': [0.06315639803617487, 0.08700484164091785],
    'lr': [0.002953296290952476, 0.005599919411324668],
    'weight_decay': [0.0001295885340230645, 0.00016240741640480654],
    'epochs': [37, 30],
    'sample_period': [0.04, 0.02],
    'seq_len': [512, 1024],
    'downsample_ratio': [1, 2] # What is this used for?
}

    # lstm_params = {
    #     'hidden_size': 32,
    #     'dropout': 0.137579431603837,
    #     'channels': [9],
    #     'lr': 0.00239595953758425,
    #     'weight_decay': 0.00016730652977231463,
    #     'epochs': 38,
    # }
    if model == 'lstm':
        run_lstm(
            train_loader,
            val_loader,
            downsample,
            SAVE_PATH,
            writer,
            enable_checkpoints,
            lstm_params,
        )
    elif model == 'cnn-lstm':
        run_cnnlstm(
            train_loader,
            val_loader,
            downsample,
            SAVE_PATH,
            writer,
            enable_checkpoints,
            lstm_params,
        )
    elif model == 'transformer':
        pass
    elif model == 'cnn-transformer':
        pass

@app.command()
def tune(
    data_json: Path,
    save_dir: Path,
    model: str, # lstm, cnn-lstm, transformer, or cnn-transformer
    test_model: bool = False, # run test after training
    enable_checkpoints: bool = False,
):
    print('test')

app()