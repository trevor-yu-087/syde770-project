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
from model.Transformer import TransformerModel
from utils.train import Transformer_train_fn
from utils.dataset import (
    SmartwatchDataset, 
    SmartwatchAugmentTransformer, 
    get_file_lists
)
from utils.utils import test_Transformer
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print(f"CUDA VISIBLE DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

# Get .csv files
train_files, val_files, test_files = get_file_lists(
    val_sub_list=['05', 10, 15, 20, 25, 30],
    test_sub_list=[35],
)

# Get dataloaders
train_dataset = SmartwatchDataset(train_files)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hp.TRANSFORMER_BATCH_SIZE, collate_fn=SmartwatchAugmentTransformer(max_input_samples=1024, downsample_output_seq=2), drop_last=True, shuffle=True)

val_dataset = SmartwatchDataset(val_files)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hp.TRANSFORMER_BATCH_SIZE, collate_fn=SmartwatchAugmentTransformer(max_input_samples=1024, downsample_output_seq=2), drop_last=True, shuffle=True)

test_dataset = SmartwatchDataset(test_files)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hp.TRANSFORMER_BATCH_SIZE, collate_fn=SmartwatchAugmentTransformer(max_input_samples=1024, downsample_output_seq=2), drop_last=True, shuffle=False)

def run(params=None,):
    SAVE_PATH = Path(f'outputs/cnntransformer_tuning_2/{datetime.now().strftime("%d-%m-%Y_%H%M%S")}')
    writer = SummaryWriter(log_dir=f'{SAVE_PATH}/tensorboard')
    TEST_PATH = SAVE_PATH

    # Initialize encoder and decoder
    transformer_model = TransformerModel(
        input_size=9,
        d_model=params['hidden_size'],
        dropout=params['dropout_p'],
        n_heads=int(params['hidden_size']/16),
        stride=2,
        kernel_size=params['kernel_size'],
        seq_len=1024,
        downsample=True,
        output_size=7,
        num_encoder_layers=params['num_layers'],
        num_decoder_layers=params['num_layers']
    ).to(hp.DEVICE)

    # Initialize optimizers
    if params:
        print(params)
        transformer_optimizer = getattr(optim, params['optimizer'])(transformer_model.parameters(), lr= params['learning_rate'], weight_decay=params['weight_decay'])
    else:
        transformer_optimizer = torch.optim.Adam(transformer_model.parameters(), lr=hp.TRANSFORMER_LEARNING_RATE)
        
    loss_fn = nn.MSELoss()
    metric_loss_fn = nn.L1Loss()

    val_loss_values = Transformer_train_fn(
            train_loader,
            val_loader,
            transformer_model,
            transformer_optimizer,
            loss_fn,
            metric_loss_fn,
            params['num_epochs'],
            hp.DEVICE,
            SAVE_PATH,
            writer,
            hp.TRANSFORMER_TEACHER_FORCE_RATIO,
            checkpoint=None,
            batch_size=hp.TRANSFORMER_BATCH_SIZE
        )
    
    return val_loss_values[-1]

def objective(trial):
    params = {
        'hidden_size': trial.suggest_categorical('hidden_size', [32, 64]),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'dropout_p': trial.suggest_float('dropout_p', 0.05, 0.15),
        'kernel_size': trial.suggest_categorical('kernel_size', [7, 15, 31, 63]),
        'optimizer': trial.suggest_categorical('optimizer', ['Adam']),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-4, 1e-2),
        'num_epochs': trial.suggest_int('num_epoch', 30, 50),
    }

    accuracy = run(params)
    return accuracy

def main():
   study =  optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
   study.optimize(objective, n_trials=100)
   print("Number of finished trials: {}".format(len(study.trials)))

   best_trial = study.best_trial
   print(f"Best trial: {best_trial}")
   for key, value in best_trial.params.items():
      print("{}: {}".format(key, value))

if __name__=='__main__':
    main()