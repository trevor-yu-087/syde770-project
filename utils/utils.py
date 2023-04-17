import os
import numpy as np
import torch
from glob import glob
import matplotlib.pyplot as plt

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
            outputs.append(test_decoder_output.numpy(force=True))
            if test_step < 1:
                plot(test_decoder_output.numpy(force=True), test_target_unpacked.numpy(force=True), test_step)

            test_loss = loss_fn(test_decoder_output, test_target_unpacked)

            # test loss
            final_test_loss += test_loss.item()

            # test metric loss
            test_metric = metric_loss_fn(test_decoder_output, test_target_unpacked)
            final_test_metric += test_metric

    print(f'Test Loss: {final_test_loss/(test_step+1)}\nTest Metric: {final_test_metric/(test_step+1)}')
    np.save(f'{path}/outputs.npy', np.array(outputs, dtype=object), allow_pickle=True)


def test_Transformer(
        test_loader,
        transformer_model,
        loss_fn,
        metric_loss_fn,
        path,
        device,
    ):
    outputs = []

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
            outputs.append(test_output.numpy(force=True))
            if test_step < 1:
                plot(test_output.numpy(force=True), test_target.numpy(force=True), test_step)

            test_loss = loss_fn(test_output, test_target)

            # test loss
            final_test_loss += test_loss.item()

            # test metric loss
            test_metric = metric_loss_fn(test_output, test_target)
            final_test_metric += test_metric

    print(f'Test Loss: {final_test_loss/(test_step+1)}\nTest Metric: {final_test_metric/(test_step+1)}')
    np.save(f'{path}/outputs.npy', np.array(outputs, dtype=object), allow_pickle=True)