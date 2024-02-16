import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
import optuna

import model.hyperparameters as hp
from model.ResNet import ResNet18_1D, ResNet_1D
from model.seq2seq_LSTM import Decoder, Encoder
from model.Transformer import TransformerModel
from utils.train import CNN_train_fn, LSTM_train_fn, Transformer_train_fn
from utils.get_loader import get_loaders

# USER EDITABLE GLOBAL VARIABLES 
###############################################################################
DATA_JSON = Path('D:\\Jonathan\\3-Datasets\\syde770_processed_data\\subjects_2023-07-12\\data.json')
DIR = Path('D:\\Jonathan\\2-Projects\\syde770-project')
MODEL = 'transformer'
###############################################################################

SAVE_DIR = Path(f'{DIR}/outputs/tuning/{MODEL}/{datetime.now().strftime("%Y-%m-%d_%H%M%S")}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Tuning {MODEL}')

train_loader, val_loader, test_loader, downsample = get_loaders(DATA_JSON, MODEL)

def objective_lstm(trial):
    params = {
        'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128]),
        'num_layers': trial.suggest_int('num_layers', 1, 5),
        'dropout_p': trial.suggest_float('dropout_p', 0.05, 0.15),
        'kernel_size': trial.suggest_categorical('kernel_size', [7, 15, 31, 63]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True),
        'num_epochs': trial.suggest_int('num_epoch', 50, 150, step=10),
        'teacher_force_ratio': trial.suggest_float('teacher_force_ratio', 0.3, 1.0, step=0.1),
        # 'dynamic_tf': trial.suggest_categorical('dynamic_tf', [True, False])
    }

    accuracy = tune_lstm(trial, params, train_loader, val_loader, downsample)
    return accuracy

def tune_lstm(trial, params, train_loader, val_loader, downsample):
    from torch.utils.tensorboard import SummaryWriter
    save_path = Path(f'{SAVE_DIR}/trial_{trial.number}')
    writer = SummaryWriter(log_dir=f'{save_path}/tensorboard')

     # Initialize encoder and decoder
    encoder_model = Encoder(
        input_size=9,
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout_p=params['dropout_p'],
        # channels=params['channels'],
        channels=9,
        stride=2,
        kernel_size=params['kernel_size'],
        seq_len=1024, # if downsample=True
        downsample=downsample,
        bidirection=False
    ).to(hp.DEVICE)
    decoder_model = Decoder(
        input_size=7,
        hidden_size=params['hidden_size'],
        output_size=7,
        num_layers=params['num_layers'],
        dropout_p=params['dropout_p'],
        bidirection=False
    ).to(hp.DEVICE)

    # Initialize loss functions
    loss_fn = nn.MSELoss()
    metric_loss_fn = nn.L1Loss()

    # Initialize optimizers
    encoder_optimizer = optim.Adam(
        encoder_model.parameters(), 
        lr=params['learning_rate'], 
        weight_decay=params['weight_decay']
        )
    decoder_optimizer = optim.Adam(
        decoder_model.parameters(), 
        lr=params['learning_rate'], 
        weight_decay=params['weight_decay']
        )

    # train
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
        save_path,
        writer,
        params['teacher_force_ratio'],
        dynamic_tf=False,
        enable_checkpoints=True,
        checkpoint=None,
    )
    return val_loss_values[-1]

def objective_transformer(trial):
    params = {
        'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128]),
        'num_layers': trial.suggest_int('num_layers', 5, 10),
        'dropout_p': trial.suggest_float('dropout_p', 0.05, 0.15),
        'kernel_size': trial.suggest_categorical('kernel_size', [7, 15, 31, 63]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True),
        'num_epochs': trial.suggest_int('num_epoch', 50, 100),
        'teacher_force_ratio': trial.suggest_float('teacher_force_ratio', 0.3, 1.0, step=0.1),
        # 'dynamic_tf': trial.suggest_categorical('dynamic_tf', [True, False])
    }

    accuracy =  tune_transformer(trial, params, train_loader, val_loader, downsample)
    return accuracy

def tune_transformer(trial, params, train_loader, val_loader, downsample):
    from torch.utils.tensorboard import SummaryWriter
    save_path = Path(f'{SAVE_DIR}/trial_{trial.number}')
    writer = SummaryWriter(log_dir=f'{save_path}/tensorboard')

    # Initialize transformer
    transformer_model = TransformerModel(
        input_size=9,
        d_model=params['hidden_size'],
        dropout=params['dropout_p'],
        n_heads=int(params['hidden_size']/4),
        stride=2,
        kernel_size=params['kernel_size'],
        seq_len=512,
        downsample=downsample,
        output_size=7,
        num_encoder_layers=params['num_layers'],
        num_decoder_layers=params['num_layers']
    ).to(hp.DEVICE)

    # Initialize loss functions
    loss_fn = nn.MSELoss()
    metric_loss_fn = nn.L1Loss()

     # Initialize optimizers
    transformer_optimizer = optim.Adam(
        transformer_model.parameters(), 
        lr=params['learning_rate'], 
        weight_decay=params['weight_decay']
    )

    val_loss_values = Transformer_train_fn(
        train_loader,
        val_loader,
        transformer_model,
        transformer_optimizer,
        loss_fn,
        metric_loss_fn,
        params['num_epochs'],
        hp.DEVICE,
        save_path,
        writer,
        params['teacher_force_ratio'],
        dynamic_tf=False,
        enable_checkpoints=True,
        checkpoint=None,
    )
    return val_loss_values[-1]
    

def main():
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(),
        study_name=MODEL
    )

    if MODEL == 'lstm':
        study.optimize(objective_lstm, n_trials=100)
    if MODEL == 'transformer':
        study.optimize(objective_transformer, n_trials=100)
    print("Number of finished trials: {}".format(len(study.trials)))

    best_trial = study.best_trial
    print(f"Best trial: {best_trial}")
    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))

if __name__ == '__main__':
    main()