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
from utils.get_loader import get_loaders
from utils.utils import (
    run_cnn,
    run_lstm, 
    run_cnnlstm, 
    run_transformer, 
    run_cnntransformer,
    test_ronin, 
    test_LSTM,
    test_transformer,
)

app = typer.Typer()

cnn_params = {
        'input_size': 32,
        'dropout': 0.1,
        'channels': 9, 
        'lr': 1e-2,
        'weight_decay': 1e-4,
        'epochs': 100,
    }

lstm_params = {
    'hidden_size': [128, 64],
    'num_layers': [2, 1],
    'kernel_size': [63, 31],
    'dropout': [0.1055526565998549, 0.1],
    'channels': [9, 64],
    'lr': [0.001146595975274196, 1e-3],
    'weight_decay': [6.889456501068434e-05, 0.000100786933714903564],
    'epochs': [100, 35],
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

@app.command()
def run(
    data_json: Path,
    save_dir: Path,
    model: str, # ronin, cnn, lstm, cnn-lstm, transformer, or cnn-transformer
    teacher_force_ratio: float = 0.5, 
    dynamic_tf: bool = False, 
    min_tf_ratio: float = 0.5,
    tf_decay: float = 0.01,
    enable_checkpoints: bool = False, 
):

    # create output save path
    SAVE_PATH = Path(f'{save_dir}/outputs/{model}/{datetime.now().strftime("%Y-%m-%d_%H%M%S")}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'Running {model} \nSave path: {SAVE_PATH}')

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=f'{SAVE_PATH}/tensorboard')
    TEST_PATH = SAVE_PATH

    train_loader, val_loader, test_loader, downsample = get_loaders(data_json, model)

    if model == 'cnn':
        run_cnn(
            train_loader,
            val_loader,
            SAVE_PATH,
            writer,
            enable_checkpoints,
            cnn_params,
        )

    if model == 'ronin':
        run_cnn(
            train_loader,
            val_loader,
            SAVE_PATH,
            writer,
            enable_checkpoints,
            cnn_params,
        )

    if model == 'lstm':
        run_lstm(
                train_loader,
                val_loader,
                downsample,
                SAVE_PATH,
                teacher_force_ratio,
                dynamic_tf,
                tf_decay,
                min_tf_ratio,
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
        run_transformer(
            train_loader,
            val_loader,
            downsample,
            SAVE_PATH,
            writer,
            enable_checkpoints,
            transformer_params,
        )
    elif model == 'cnn-transformer':
        run_cnntransformer(
            train_loader,
            val_loader,
            downsample,
            SAVE_PATH,
            writer,
            enable_checkpoints,
            transformer_params,
        )      

@app.command()
def test(
    data_json: Path,
    model: str, # ronin, cnn, lstm, cnn-lstm, transformer, or cnn-transformer
    checkpoint_path: str = None, # path to model checkpoints   
):
    _, _, test_loader, _ = get_loaders(data_json, model)

    with data_json.open('r') as f:
        valid_files = json.loads(f.read())
    f.close()
    test_files = valid_files['test']

    if model == 'ronin':
            test_ronin(
                test_files,
                checkpoint_path,
            )
    elif model == 'lstm':
        test_LSTM(
            test_loader,
            lstm_params,
            checkpoint_path
        )
    elif model == 'cnn-lstm':
        test_LSTM(
            test_loader,
            lstm_params,
            checkpoint_path,
            downsample = True
        )
    else:
        raise Exception('Unsupported model type')


app()