import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
from datetime import datetime
import typer
from enum import Enum

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
class NeuralNetwork(str, Enum):
    ronin = 'ronin'
    lstm = 'lstm'
    cnn_lstm = 'cnn_lstm'
    transformer = 'transformer'
    cnn_transformer = 'cnn_transformer'

# parameters for training/testing
cnn_params = {
        'input_size': 32,
        'dropout': 0.1,
        'channels': 9, 
        'lr': 1e-2,
        'weight_decay': 1e-4,
        'epochs': 100,
    }
lstm_params = {
    'hidden_size': [64, 64],
    'num_layers': [1, 1],
    'kernel_size': [63, 63],
    'dropout': [0.08698052733915022, 0.1],
    'channels': [9, 64],
    'lr': [0.007948288707394854, 1e-3],
    'weight_decay': [1.8554248638243292e-06, 0.000100786933714903564],
    'epochs': [500, 35],
}
transformer_params = {
'hidden_size': [128, 128],
'dropout': [0.06315639803617487, 0.08700484164091785],
'lr': [0.002953296290952476, 0.005599919411324668],
'weight_decay': [0.0001295885340230645, 0.00016240741640480654],
'epochs': [50, 30],
'sample_period': [0.04, 0.02],
'downsample_ratio': [1, 2] # What is this used for?
}

@app.command()
def run(
    data_json: Path,
    save_dir: Path,
    model: NeuralNetwork,
    seq_len: int=32,
    teacher_force_ratio: float = 0.5, 
    dynamic_tf: bool = False, 
    min_tf_ratio: float = 0.5,
    tf_decay: float = 0.01,
    enable_checkpoints: bool = False, 
) -> None:
    """Run model training 

    Parameters:
    -----------
    data_json (Path): path to json with dataset paths (see README for examples)
    save_dir (Path): path to output directory
    model (string [ronin, lstm, cnn_lstm, transformer, cnn_transformer]): name of model for training
    seq_len (integer): sequence length for input (note sequence length output is halved for cnn downsampling models)
    teacher_force_ratio (float [0.0:1.0]): teacher force ratio for lstm/transformer variant models
    dynamic_tf (bool): use of a dynamic teacher force ratio (ie. tf ratio decay)
    min_tf_ratio (float): minimum teacher force ratio value can decay to
    tf_decay (float): teacher force ratio decay value subtracted each step
    enable_checkpoints (bool): save epoch checkpoints
    """
    # create output save path
    SAVE_PATH = Path(f'{save_dir}/outputs/{model}/{datetime.now().strftime("%Y-%m-%d_%H%M%S")}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'Running {model} \nSave path: {SAVE_PATH}')

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=f'{SAVE_PATH}/tensorboard')

    # get loaders
    train_loader, val_loader, test_loader, downsample = get_loaders(data_json, model, seq_len)

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
                train_loader,
                seq_len,
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
    elif model == 'cnn_lstm':
        run_cnnlstm(
            train_loader,
            val_loader,
            seq_len,
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
    elif model == 'transformer':
        run_transformer(
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
            transformer_params,
        )
    elif model == 'cnn_transformer':
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
    checkpoint_path: Path,
    model: str, 
    seq_len: int=32,  
):
    """Run model testing

    Parameters:
    -----------
    data_json (Path): path to json with dataset paths (see README for examples)
    checkpoint_path (Path): path to checkpoints directory output from training
    model (string [ronin, lstm, cnn_lstm, transformer, cnn_transformer]): name of model for training
    seq_len (integer): sequence length for input (note sequence length output is halved for cnn downsampling models)
    """
    _, _, test_loader, _ = get_loaders(data_json, model, seq_len)

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
            checkpoint_path,
            seq_len
        )
    elif model == 'cnn_lstm':
        test_LSTM(
            test_loader,
            lstm_params,
            checkpoint_path,
            seq_len,
            downsample = True
        )
    elif model == 'transformer':
        test_transformer(
            test_loader,
            transformer_params,
            checkpoint_path,
            seq_len
        )
    else:
        raise Exception('Unsupported model type')


app()