import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import os
import glob
from datetime import datetime

import model.hyperparameters as hp
from model.Transformer import TransformerModel
from utils.train import Transformer_train_fn
from utils.dataset import (
    SmartwatchDataset, 
    SmartwatchAugmentTransformer, 
    get_file_lists
)
from utils.utils import test_Transformer

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# print(f"CUDA VISIBLE DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

BATCH_SIZE = 1

# Cross validation stuff
START_SUBJECT = 6
val_subjects = []
for i in range(START_SUBJECT, (START_SUBJECT+30), 5):
     val_subjects.append(i)

TRANSFORMER_OR_CNNTRANSFORMER = 0
params = {
    'hidden_size': [32, 128],
    'dropout': [0.06315639803617487, 0.08700484164091785],
    'downsample': [False, True],
    'lr': [0.002953296290952476, 0.005599919411324668],
    'weight_decay': [0.0001295885340230645, 0.00016240741640480654],
    'epochs': [37, 30],
    'transformer_or_cnntransformer': ['Transformer', 'CNN-Transformer'],
    'sample_period': [0.04, 0.02],
    'seq_len': [512, 1024],
    'downsample_ratio': [1, 2]
}

# Paths
SAVE_PATH = Path(f'outputs/cross-val/{params["transformer_or_cnntransformer"][TRANSFORMER_OR_CNNTRANSFORMER]}/S{START_SUBJECT}-5-{val_subjects[-1]}/{datetime.now().strftime("%d-%m-%Y_%H%M%S")}')
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
        val_subjects,
        test_sub_list=[41],
    )

    # Get dataloaders
    train_dataset = SmartwatchDataset(train_files, params['sample_period'][TRANSFORMER_OR_CNNTRANSFORMER])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=SmartwatchAugmentTransformer(max_input_samples=params['seq_len'][TRANSFORMER_OR_CNNTRANSFORMER], downsample_output_seq=params['downsample_ratio'][TRANSFORMER_OR_CNNTRANSFORMER]), drop_last=True, shuffle=True)

    val_dataset = SmartwatchDataset(val_files, params['sample_period'][TRANSFORMER_OR_CNNTRANSFORMER])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=SmartwatchAugmentTransformer(max_input_samples=params['seq_len'][TRANSFORMER_OR_CNNTRANSFORMER], downsample_output_seq=params['downsample_ratio'][TRANSFORMER_OR_CNNTRANSFORMER]), drop_last=True, shuffle=True)

    test_dataset = SmartwatchDataset(test_files, params['sample_period'][TRANSFORMER_OR_CNNTRANSFORMER])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=SmartwatchAugmentTransformer(max_input_samples=params['seq_len'][TRANSFORMER_OR_CNNTRANSFORMER], downsample_output_seq=params['downsample_ratio'][TRANSFORMER_OR_CNNTRANSFORMER]), drop_last=True, shuffle=False)

    # Initialize transformer
    transformer_model = TransformerModel(
        input_size=9,
        d_model=params['hidden_size'][TRANSFORMER_OR_CNNTRANSFORMER],
        dropout=params['dropout'][TRANSFORMER_OR_CNNTRANSFORMER],
        n_heads=int(params['hidden_size'][TRANSFORMER_OR_CNNTRANSFORMER]/4),
        stride=2,
        kernel_size=15,
        seq_len=params['seq_len'][TRANSFORMER_OR_CNNTRANSFORMER],
        downsample=params['downsample'][TRANSFORMER_OR_CNNTRANSFORMER],
        output_size=7,
        num_encoder_layers=5,
        num_decoder_layers=5
    ).to(hp.DEVICE)

    # Initialize loss functions
    loss_fn = nn.MSELoss()
    metric_loss_fn = nn.L1Loss()

    # Initialize optimizers
    transformer_optimizer = optim.Adam(transformer_model.parameters(), lr=params['lr'][TRANSFORMER_OR_CNNTRANSFORMER], weight_decay=params['weight_decay'][TRANSFORMER_OR_CNNTRANSFORMER])

    if TRAIN == True:
        Transformer_train_fn(
            train_loader,
            val_loader,
            transformer_model,
            transformer_optimizer,
            loss_fn,
            metric_loss_fn,
            params['epochs'][TRANSFORMER_OR_CNNTRANSFORMER],
            hp.DEVICE,
            SAVE_PATH,
            writer,
            hp.TEACHER_FORCE_RATIO,
            checkpoint=None,
            batch_size=BATCH_SIZE
        )

    # test_Transformer(
    #     test_loader,
    #     transformer_model,
    #     loss_fn,
    #     metric_loss_fn,
    #     SAVE_PATH,
    #     hp.DEVICE,
    # )

if __name__ == '__main__':
    main()