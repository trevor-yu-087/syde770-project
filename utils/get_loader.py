import torch
import json

import model.hyperparameters as hp
from utils.dataset import (
    SmartwatchDataset, 
    SmartwatchAugmentCnn,
    SmartwatchAugmentRonin,
    SmartwatchAugmentLstm, 
    SmartwatchAugmentTransformer, 
)
def get_loaders(data_json, model, seq_len):
    # parameters for models
    if model == 'lstm' or model =='transformer' or model == 'ronin':
        sample_period = 0.04
        downsample = False
        max_samples = seq_len
        if model == 'ronin':
            max_samples = 32
    elif model == 'cnn_lstm' or model == 'cnn_transformer':
        sample_period = 0.02
        downsample = True
        max_samples = seq_len * 2
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
        collate_fn = SmartwatchAugmentRonin(max_input_samples=32)
        test_collate_fn = SmartwatchAugmentRonin(max_input_samples=32, augment=False)
    elif model == 'lstm' or model == 'cnn_lstm':
        collate_fn = SmartwatchAugmentLstm(max_input_samples=max_samples)
        test_collate_fn = SmartwatchAugmentLstm(max_input_samples=max_samples, augment=False)
    elif model == 'transformer':
        collate_fn = SmartwatchAugmentTransformer(
            max_input_samples=max_samples, 
            downsample_output_seq=1
        )
        test_collate_fn = SmartwatchAugmentTransformer(
            max_input_samples=max_samples, 
            downsample_output_seq=1,
            augment=False
        )
    elif model == 'cnn_transformer':
        collate_fn = SmartwatchAugmentTransformer(
            max_input_samples=max_samples, 
            downsample_output_seq=2
        )
        test_collate_fn = SmartwatchAugmentTransformer(
            max_input_samples=max_samples, 
            downsample_output_seq=2,
            augment=False
        )
    
    train_dataset = SmartwatchDataset(train_files, sample_period)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hp.BATCH_SIZE,
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=True,
        pin_memory=True,
        # num_workers=hp.NUM_WORKERS,
        # persistent_workers=True
    )

    val_dataset = SmartwatchDataset(val_files, sample_period)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=hp.BATCH_SIZE,
        collate_fn=test_collate_fn,
        drop_last=True,
        shuffle=True,
        pin_memory=True,
        # num_workers=hp.NUM_WORKERS,
        # persistent_workers=True
    )

    test_dataset = SmartwatchDataset(test_files, sample_period)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=hp.BATCH_SIZE,
        collate_fn=test_collate_fn,
        drop_last=True,
        shuffle=False,
        pin_memory=True,
        # num_workers=hp.NUM_WORKERS,
        # persistent_workers=True
    )

    return train_loader, val_loader, test_loader, downsample