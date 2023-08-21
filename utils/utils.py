import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import model.hyperparameters as hp
from model.ResNet import ResNet18_1D, ResNet_1D
from model.seq2seq_LSTM import Decoder, Encoder
from model.Transformer import TransformerModel
from utils.train import CNN_train_fn, LSTM_train_fn, Transformer_train_fn
from utils.visualize import pred_vs_error
from utils.dataset import get_ronin_data
from utils.metric import compute_ate_rte


def run_cnn (train_loader, val_loader, save_path, writer, enable_checkpoints, params = None):
    # initialize 1D ResNet18
    model = ResNet_1D(num_classes=3).to(hp.DEVICE)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)

    # Initialize loss functions
    loss_fn = nn.MSELoss()
    metric_loss_fn = nn.L1Loss()

    # Initialize optimizers
    optimizer = optim.Adam(
        model.parameters(), 
        lr=params['lr'], 
        weight_decay=params['weight_decay']
        )

    # train
    _ = CNN_train_fn(
        train_loader,
        val_loader,
        model,
        optimizer,
        loss_fn,
        metric_loss_fn,
        params['epochs'],
        hp.DEVICE,
        save_path,
        writer,
        enable_checkpoints,
        checkpoint=None,
    )

def run_lstm (train_loader, val_loader, downsample, save_path, writer, enable_checkpoints, params = None):
    # Initialize encoder and decoder
    encoder_model = Encoder(
        input_size=9,
        hidden_size=params['hidden_size'][0],
        num_layers=1,
        dropout_p=params['dropout'][0],
        channels=params['channels'][0],
        stride=2,
        kernel_size=63,
        seq_len=1024,
        downsample=downsample,
    ).to(hp.DEVICE)
    decoder_model = Decoder(
        input_size=7,
        hidden_size=params['hidden_size'][0],
        output_size=7,
        num_layers=1,
        dropout_p=params['dropout'][0],
    ).to(hp.DEVICE)

    # Initialize loss functions
    loss_fn = nn.MSELoss()
    metric_loss_fn = nn.L1Loss()

    # Initialize optimizers
    encoder_optimizer = optim.Adam(
        encoder_model.parameters(), 
        lr=params['lr'][0], 
        weight_decay=params['weight_decay'][0]
        )
    decoder_optimizer = optim.Adam(
        decoder_model.parameters(), 
        lr=params['lr'][0], 
        weight_decay=params['weight_decay'][0]
        )

    # train
    _ = LSTM_train_fn(
        train_loader,
        val_loader,
        encoder_model,
        decoder_model,
        encoder_optimizer,
        decoder_optimizer,
        loss_fn,
        metric_loss_fn,
        params['epochs'][0],
        hp.DEVICE,
        save_path,
        writer,
        hp.TEACHER_FORCE_RATIO,
        enable_checkpoints,
        checkpoint=None,
    )

def run_cnnlstm (train_loader, val_loader, downsample, save_path, writer, enable_checkpoints, params = None):
    # Initialize encoder and decoder
    encoder_model = Encoder(
        input_size=9,
        hidden_size=params['hidden_size'][1],
        num_layers=1,
        dropout_p=params['dropout'][1],
        channels=params['channels'][1],
        stride=2,
        kernel_size=63,
        seq_len=1024,
        downsample=downsample,
    ).to(hp.DEVICE)
    decoder_model = Decoder(
        input_size=7,
        hidden_size=params['hidden_size'][1],
        output_size=7,
        num_layers=1,
        dropout_p=params['dropout'][1],
    ).to(hp.DEVICE)

    # Initialize loss functions
    loss_fn = nn.MSELoss()
    metric_loss_fn = nn.L1Loss()

    # Initialize optimizers
    encoder_optimizer = optim.Adam(
        encoder_model.parameters(), 
        lr=params['lr'][1], 
        # weight_decay=params['weight_decay'][1]
    )
    decoder_optimizer = optim.Adam(
        decoder_model.parameters(), 
        lr=params['lr'][1], 
        # weight_decay=params['weight_decay'][1]
    )

    # train
    _ = LSTM_train_fn(
        train_loader,
        val_loader,
        encoder_model,
        decoder_model,
        encoder_optimizer,
        decoder_optimizer,
        loss_fn,
        metric_loss_fn,
        params['epochs'][1],
        hp.DEVICE,
        save_path,
        writer,
        hp.TEACHER_FORCE_RATIO,
        enable_checkpoints,
        checkpoint=None,
    )

def run_transformer(train_loader, val_loader, downsample, save_path, writer, enable_checkpoints, params = None):
    # Initialize transformer
    transformer_model = TransformerModel(
        input_size=9,
        d_model=params['hidden_size'][0],
        dropout=params['dropout'][0],
        n_heads=int(params['hidden_size'][0]/4),
        stride=2,
        kernel_size=15,
        seq_len=params['seq_len'][0],
        downsample=downsample,
        output_size=7,
        num_encoder_layers=5,
        num_decoder_layers=5
    ).to(hp.DEVICE)

    # pytorch_total_params = sum(p.numel() for p in transformer_model.parameters())
    # print(f"Model params: {pytorch_total_params}")

    # Initialize loss functions
    loss_fn = nn.MSELoss()
    metric_loss_fn = nn.L1Loss()

    # Initialize optimizers
    transformer_optimizer = optim.Adam(
        transformer_model.parameters(), 
        lr=params['lr'][0], 
        weight_decay=params['weight_decay'][0]
    )

    Transformer_train_fn(
        train_loader,
        val_loader,
        transformer_model,
        transformer_optimizer,
        loss_fn,
        metric_loss_fn,
        params['epochs'][0],
        hp.DEVICE,
        save_path,
        writer,
        hp.TRANSFORMER_TEACHER_FORCE_RATIO,
        enable_checkpoints,
        checkpoint=None,
        batch_size=hp.BATCH_SIZE,
    )

def run_cnntransformer(train_loader, val_loader, downsample, save_path, writer, enable_checkpoints, params = None):
        # Initialize transformer
    transformer_model = TransformerModel(
        input_size=9,
        d_model=params['hidden_size'][1],
        dropout=params['dropout'][1],
        n_heads=int(params['hidden_size'][1]/4),
        stride=2,
        kernel_size=15,
        seq_len=params['seq_len'][1],
        downsample=downsample,
        output_size=7,
        num_encoder_layers=5,
        num_decoder_layers=5
    ).to(hp.DEVICE)

    # pytorch_total_params = sum(p.numel() for p in transformer_model.parameters())
    # print(f"Model params: {pytorch_total_params}")

    # Initialize loss functions
    loss_fn = nn.MSELoss()
    metric_loss_fn = nn.L1Loss()

    # Initialize optimizers
    transformer_optimizer = optim.Adam(
        transformer_model.parameters(), 
        lr=params['lr'][1], 
        weight_decay=params['weight_decay'][1]
    )

    Transformer_train_fn(
        train_loader,
        val_loader,
        transformer_model,
        transformer_optimizer,
        loss_fn,
        metric_loss_fn,
        params['epochs'][1],
        hp.DEVICE,
        save_path,
        writer,
        hp.TRANSFORMER_TEACHER_FORCE_RATIO,
        enable_checkpoints,
        checkpoint=None,
        batch_size=hp.BATCH_SIZE,
    )

def load_checkpoint(encoder_model, decoder_model, path):
    """Load checkpoint into LSTM test model
    Parameters:
    -----------
    encoder_model: initialized LSTM encoder model
    decoder_model: initialized LSTM decoder model
    path: path to trained weights
    Returns:
    --------
    encoder_model: LSTM encoder model with loaded weights
    decoder_model: LSTM decoder model with loaded weights
    """
    encoder_model.load_state_dict(torch.load(
        os.path.join(path, 'best', 'best_encoder_model.pth'),
        map_location='cpu'
    ))
    decoder_model.load_state_dict(torch.load(
        os.path.join(path, 'best', 'best_decoder_model.pth'),
        map_location='cpu'
    ))
    return encoder_model, decoder_model

def load_checkpoint_Transformer(transformer_model, path):
    transformer_model.load_state_dict(torch.load(
        os.path.join(path, 'best', 'best_transformer_model.pth'),
        map_location='cpu'
    ))
    
    return transformer_model

# def plot(data, title):
#     for i in range(data.shape[0]):
#         fig, axs = plt.subplots(2, 1, figsize=(10, 6))
#         labels = ["X", "Y", "Z", "qx", "qy", "qz", "qw"]
#         b, l, d = data.shape
#         t = np.arange(512)
#         for j in range(3):
#             axs[0].plot(t, data[i, :, j], label=labels[j])
#         axs[0].legend()
#         for j in range(3, d):
#             axs[1].plot(t, data[i, :, j], label=labels[j])
#         axs[1].legend()
#         plt.suptitle(f'{title} Batch {i}')
#         plt.show()

def plot(outputs, targets, step):
    for i in range(targets.shape[0]):
        fig, axs = plt.subplots(4, 1, figsize=(10, 12))
        labels = ["X", "Y", "Z", "qx", "qy", "qz", "qw"]
        b, l, d = targets.shape
        t = np.arange(512)
        for j in range(3):
            axs[0].plot(t, outputs[i, :, j], label=labels[j])
            axs[0].set_title(f'Test Output: Step {step} Batch {i}')
            axs[1].plot(t, targets[i, :, j], label=labels[j])
            axs[1].set_title(f'Test Target: Step {step} Batch {i}')
        axs[0].legend()
        axs[1].legend()
        for j in range(3, d):
            axs[2].plot(t, outputs[i, :, j], label=labels[j])
            axs[2].set_title(f'Test Output: Step {step} Batch {i}')
            axs[3].plot(t, targets[i, :, j], label=labels[j])
            axs[3].set_title(f'Test Target: Step {step} Batch {i}')
        axs[1].legend()
        # plt.suptitle(f'{title} Batch {i}')
        plt.show()

def test_ronin(
        test_files,
        checkpoint_path,
):
    # initialize ResNet
    model = ResNet_1D(num_classes=3).to(hp.DEVICE)
    model.load_state_dict(torch.load(
        os.path.join(checkpoint_path, 'best', 'best_model.pth'),
        map_location='cpu'
    ))

    # Initialize loss functions
    loss_fn = nn.MSELoss()
    metric_loss_fn = nn.L1Loss()

    # get velocity stats
    x, y, z = [], [], []
    for file in(test_files):
        _, targets, _, _ = get_ronin_data(file)
        x.append(targets[:, 0].numpy())
        y.append(targets[:, 1].numpy())
        z.append(targets[:, 2].numpy())

    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    z = np.concatenate(z, axis=0)
    velocity_std = [
        np.std(x),
        np.std(y),
        np.std(z)
    ]
    velocity_mean = [
        np.mean(x),
        np.mean(y),
        np.mean(z)
    ]

    model.eval()
    with torch.no_grad():
        final_test_loss = 0
        final_test_metric = 0
        final_ate = 0
        final_rte = 0

        preds, targets = [], []

        for num_file, test_file in enumerate(test_files):
            test_source, test_target, test_target_pos, smooth_pos = get_ronin_data(test_file)

            # zero-score normalize velocity targets
            for i in range(3):
                test_target[:,i] = (test_target[:,i] - velocity_mean[i]) / velocity_std[i]

            test_source = test_source.to(hp.DEVICE)
            test_target = test_target.to(hp.DEVICE)

            # Run validation model
            test_pred = model(test_source)
            
            test_loss = loss_fn(test_pred, test_target)

            # test loss
            final_test_loss += test_loss.item()

            # test metric loss
            test_metric = metric_loss_fn(test_pred, test_target)
            final_test_metric += test_metric

            # calculate ate and rte
            ate, rte = compute_ate_rte(
                test_pred.cpu().detach().numpy(), 
                test_target.cpu().detach().numpy(), 
                pred_per_min=50*10
            )
            final_ate += ate
            final_rte += rte
            print(f'File: {num_file} \tATE: {ate} \tRTE: {rte}')

            # undo zero-score normalization to velocity 
            for i in range(3):
                test_pred[:,i] = (test_pred[:,i] * velocity_std[i]) + velocity_mean[i]
                test_target[:,i] = (test_target[:,i] * velocity_std[i]) + velocity_mean[i]

            # append un-normalized pred and targets for visualization
            preds.append(test_pred.cpu().detach().numpy())
            targets.append(test_target.cpu().detach().numpy())

            # get starting position seed from ground truth for each batch
            start_pos = np.zeros((test_target.shape))
            start_pos[0,:] = test_target_pos[0,:,0]

            # change in position from test predictions
            ds = (test_pred * (0.02*32)).cpu().detach().numpy()
            pos = start_pos + ds
            pos = np.cumsum(pos, axis=0)

            plot_paths = True
            if plot_paths and (num_file == 4):
                fig, axes = plt.subplots(3, 1, tight_layout=True)
                fig.suptitle(test_file)
                x = np.arange(31, (pos.shape[0]*32), 32) 
                axes[0].set_title("X Positions")
                axes[0].plot(x, pos[:,0])
                axes[0].plot(smooth_pos[:,0])
                axes[0].plot(start_pos[0,0], 'x')

                axes[1].set_title("Y Positions")
                axes[1].plot(x, pos[:,1])
                axes[1].plot(smooth_pos[:,1])

                axes[2].set_title("Z Positions")
                axes[2].plot(x, pos[:,2])
                axes[2].plot(smooth_pos[:,2])

                plt.show()
    
    print(f'Test Loss: {final_test_loss/(num_file+1)}\nTest Metric: {final_test_metric/(num_file+1)}')
    print(f'ATE: {final_ate/(num_file+1)}\nRTE: {final_rte/(num_file+1)}')
    pred_vs_error(preds, targets, 'RoNIN Test')

    # np.save(f'outputs.npy', np.array(outputs, dtype=object), allow_pickle=True)
    # np.save(f'targets.npy', np.array(targets, dtype=object), allow_pickle=True)



def test_LSTM(
        test_loader,
        encoder_model,
        decoder_model,
        loss_fn,
        metric_loss_fn,
        path,
        device,
    ):
    outputs = []
    targets = []

    encoder_model, decoder_model = load_checkpoint(encoder_model, decoder_model, path)
    encoder_model.eval()
    decoder_model.eval()
    with torch.no_grad():
        final_test_loss = 0
        final_test_metric = 0

        for test_step, test_data in enumerate(test_loader):
            test_source = test_data['encoder_inputs'].to(device)
            test_target = test_data['decoder_inputs'].to(device)
            test_target_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(test_target, batch_first=True)
            test_target_unpacked.to(device)

            # Run test model
            test_encoder_hidden, test_encoder_cell = encoder_model(test_source)
            test_encoder_cell = torch.zeros(test_encoder_cell.shape).to(device)

            test_decoder_output, test_decoder_hidden, test_decoder_cell = decoder_model(test_target, test_encoder_hidden, test_encoder_cell)
            # outputs.append(test_decoder_output.numpy(force=True))
            if test_step < 1:
                plot(test_decoder_output.numpy(force=True), test_target_unpacked.numpy(force=True), test_step)
                outputs.append((test_decoder_output.numpy(force=True)))
                targets.append(test_target_unpacked.numpy(force=True))

            test_loss = loss_fn(test_decoder_output, test_target_unpacked)

            # test loss
            final_test_loss += test_loss.item()

            # test metric loss
            test_metric = metric_loss_fn(test_decoder_output, test_target_unpacked)
            final_test_metric += test_metric

    print(f'Test Loss: {final_test_loss/(test_step+1)}\nTest Metric: {final_test_metric/(test_step+1)}')
    np.save(f'{path}/outputs.npy', np.array(outputs, dtype=object), allow_pickle=True)
    np.save(f'{path}/targets.npy', np.array(targets, dtype=object), allow_pickle=True)


def test_transformer(
        test_loader,
        transformer_model,
        loss_fn,
        metric_loss_fn,
        path,
        device,
    ):
    outputs = []
    targets = []

    transformer_model = load_checkpoint_Transformer(transformer_model, path)
    transformer_model.eval()
    
    with torch.no_grad():
        final_test_loss = 0
        final_test_metric = 0

        for test_step, test_data in enumerate(test_loader):
            test_source = test_data['encoder_inputs'].to(device)
            test_target = test_data['decoder_inputs'].to(device)
            test_target.to(device)

            # Run test model
            test_output = transformer_model(test_source, test_target)
            #test_encoder_cell = torch.zeros(1, 4, 32).to(device)

            #test_decoder_output, test_decoder_hidden, test_decoder_cell = decoder_model(test_target, test_encoder_hidden, test_encoder_cell)
            # outputs.append(test_output.numpy(force=True))
            if test_step < 1:
                plot(test_output.numpy(force=True), test_target.numpy(force=True), test_step)
                outputs.append((test_output.numpy(force=True)))
                targets.append(test_target.numpy(force=True))
            test_loss = loss_fn(test_output, test_target)

            # test loss
            final_test_loss += test_loss.item()

            # test metric loss
            test_metric = metric_loss_fn(test_output, test_target)
            final_test_metric += test_metric

    print(f'Test Loss: {final_test_loss/(test_step+1)}\nTest Metric: {final_test_metric/(test_step+1)}')
    np.save(f'{path}/outputs.npy', np.array(outputs, dtype=object), allow_pickle=True)
    np.save(f'{path}/targets.npy', np.array(targets, dtype=object), allow_pickle=True)