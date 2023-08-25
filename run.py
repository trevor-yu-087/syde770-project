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
    SmartwatchAugmentCnn,
    SmartwatchAugmentRonin,
    SmartwatchAugmentLstm, 
    SmartwatchAugmentTransformer, 
)
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
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'epochs': 100,
    }

lstm_params = {
    'hidden_size': [32, 64],
    'dropout': [0.137579431603837, 0.1],
    'channels': [9, 64],
    'lr': [1e-3, 1e-3],
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

def get_loaders(data_json, model):
    # parameters for models
    if model == 'lstm' or model =='transformer' or model == 'cnn' or model == 'ronin':
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

    if model == 'cnn':
        collate_fn = SmartwatchAugmentCnn()
        test_collate_fn = SmartwatchAugmentCnn(augment=False)
    if model == 'ronin':
        collate_fn = SmartwatchAugmentRonin(max_input_samples=cnn_params['input_size'])
        test_collate_fn = SmartwatchAugmentRonin(max_input_samples=cnn_params['input_size'], augment=False)
    elif model == 'lstm' or model == 'cnn-lstm':
        collate_fn = SmartwatchAugmentLstm()
        test_collate_fn = SmartwatchAugmentLstm(augment=False)
    elif model == 'transformer':
        collate_fn = SmartwatchAugmentTransformer(
            max_input_samples=transformer_params['seq_len'][0], 
            downsample_output_seq=transformer_params['downsample_ratio'][0]
        )
        test_collate_fn = SmartwatchAugmentTransformer(
            max_input_samples=transformer_params['seq_len'][0], 
            downsample_output_seq=transformer_params['downsample_ratio'][0],
            augment=False
        )
    elif model == 'cnn-transformer':
        collate_fn = SmartwatchAugmentTransformer(
            max_input_samples=transformer_params['seq_len'][1], 
            downsample_output_seq=transformer_params['downsample_ratio'][1]
        )
        test_collate_fn = SmartwatchAugmentTransformer(
            max_input_samples=transformer_params['seq_len'][1], 
            downsample_output_seq=transformer_params['downsample_ratio'][1],
            augment=False
        )
    
    train_dataset = SmartwatchDataset(train_files, sample_period)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hp.BATCH_SIZE, collate_fn=collate_fn, drop_last=True, shuffle=False)

    val_dataset = SmartwatchDataset(val_files, sample_period)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hp.BATCH_SIZE, collate_fn=test_collate_fn, drop_last=True, shuffle=False)

    test_dataset = SmartwatchDataset(test_files, sample_period)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hp.BATCH_SIZE, collate_fn=test_collate_fn, drop_last=True, shuffle=False)

    return train_loader, val_loader, test_loader, downsample

@app.command()
def run(
    data_json: Path,
    save_dir: Path,
    model: str, # ronin, cnn, lstm, cnn-lstm, transformer, or cnn-transformer
    enable_checkpoints: bool = False, 
):

    # create output save path
    SAVE_PATH = Path(f'{save_dir}/outputs/{model}/{datetime.now().strftime("%Y-%m-%d_%H%M%S")}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'Running {model} \nSave path: {SAVE_PATH}')
    print(f'RoNIN input size: {cnn_params["input_size"]}')
    print(f'RoNIN learning rate: {cnn_params["lr"]}')


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
def tune(
    data_json: Path,
    save_dir: Path,
    model: str, # lstm, cnn-lstm, transformer, or cnn-transformer
    test_model: bool = False, # run test after training
    enable_checkpoints: bool = False,
):
    print('test')

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