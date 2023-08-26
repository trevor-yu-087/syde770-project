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
def get_loaders(data_json, model, cnn_params, lstm_params, transformer_params):
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