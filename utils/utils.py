import os
import torch
from glob import glob

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

def test_LSTM(
        test_loader,
        encoder_model,
        decoder_model,
        loss_fn,
        metric_loss_fn,
        path,
        device,
    ):
    encoder_model, decoder_model = load_checkpoint(encoder_model, decoder_model, path)

    encoder_model.eval()
    decoder_model.eval()
    with torch.no_grad():
        final_test_loss = 0
        final_test_metric = 0

        for test_step, test_data in enumerate(test_loader):
            test_source = test_data['encoder_inputs'].to(device)
            test_target = test_data['decoder_inputs'].to(device)

            # Run test model
            test_encoder_hidden, test_encoder_cell = encoder_model(test_source)
            test_decoder_output, test_decoder_hidden, test_decoder_cell = decoder_model(test_target, test_encoder_hidden, test_encoder_cell)

            test_loss = loss_fn(test_decoder_output, test_target)

            # test loss
            final_test_loss += test_loss.item()

            # test metric loss
            test_metric = metric_loss_fn(test_decoder_output, test_target)
            final_test_metric += test_metric

    print(f'Test Loss: {final_test_loss/(test_step+1)}\nTest Metric: {final_test_metric/(test_step+1)}')