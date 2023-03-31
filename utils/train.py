import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os

def LSTM_train_fn(
        train_loader,
        val_loader,
        encoder_model,
        decoder_model,
        encoder_optimizer,
        decoder_optimizer,
        loss_fn,
        metric_loss_fn,
        num_epoch,
        device,
        save_path,
        writer,
        teacher_force_ratio=1,
        val_interval=1,
        checkpoint=None,
):
    for epoch in range(num_epoch):
        
        epoch_train_loss = 0
        epoch_train_metric = 0

        for train_step, train_data in enumerate(train_loader):
            train_source = train_data['encoder_inputs'].to(device)
            train_target = train_data['decoder_inputs'].to(device)

            # Zero optimizers
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # Forward pass
            decoder_output = torch.zeros(4, train_target.shape[1], train_target.shape[2])
            start = train_target[:, 0, :]
            teacher_force = True if random.random() < teacher_force_ratio else False

            encoder_hidden, _ = encoder_model(train_source)
            encoder_cell = torch.zeros(1, 4, 32)
            
            if epoch == 0:
                decoder_output, decoder_hidden, decoder_cell = decoder_model(train_target, encoder_hidden, encoder_cell)
                print(f'Decoder Output: {decoder_output.shape}\t Decoder Hidden: {decoder_hidden.shape}\t Decoder Cell: {decoder_cell.shape}')
            elif epoch !=0 and teacher_force == True:
                decoder_output, decoder_hidden, decoder_cell = decoder_model(train_target, encoder_hidden, encoder_cell)
            elif epoch != 0 and teacher_force == False:
                for i in range(1, train_target.shape[1]):
                    decoder_output[:, i, :], decoder_hidden, decoder_cell = decoder_model(start.unsqueeze(1), encoder_hidden, encoder_cell)
                    encoder_hidden = decoder_hidden
                    encoder_cell = decoder_cell

            train_loss = loss_fn(decoder_output, train_target)

            # Backwards
            train_loss.backward()

            # Update optimizers
            encoder_optimizer.step()
            decoder_optimizer.step()

            # Train loss
            epoch_train_loss += train_loss.item()

            # Train metric loss
            train_metric = metric_loss_fn(decoder_output, train_target)
            epoch_train_metric += train_metric

        # Average losses for tensorboard
        epoch_train_loss /= (train_step+1)
        writer.add_scalar('Training MSE per Epoch', epoch_train_loss, epoch)
        epoch_train_metric /= (train_step+1)
        writer.add_scalar('Training MAE per Epoch', epoch_train_metric, epoch)
        

        if epoch % val_interval == 0:
            encoder_model.eval()
            decoder_model.eval()
            with torch.no_grad():
                epoch_val_loss = 0
                epoch_val_metric = 0

                for val_step, val_data in enumerate(val_loader):
                    val_source = val_data['encoder_inputs'].to(device)
                    val_target = val_data['decoder_inputs'].to(device)

                    # Run validation model
                    val_encoder_hidden, val_encoder_cell = encoder_model(val_source)
                    val_decoder_output, val_decoder_hidden, val_decoder_cell = decoder_model(val_target, val_encoder_hidden, val_encoder_cell)

                    val_loss = loss_fn(val_decoder_output, val_target)

                    # Val loss
                    epoch_val_loss += val_loss.item()

                    # Val metric loss
                    val_metric = metric_loss_fn(val_decoder_output, val_target)
                    epoch_val_metric += val_metric

                # Average validation losses for tensorboard
                epoch_val_loss /= (val_step+1)
                writer.add_scalar('Validation MSE per Epoch', epoch_val_loss, epoch)
                epoch_val_metric /= (val_step+1)
                writer.add_scalar('Validation MAE per Epoch', epoch_val_metric, epoch)


                 # Save checkpoint
                if not os.path.exists(os.path.join(save_path, 'checkpoint')):
                    os.makedirs(os.path.join(save_path, 'checkpoint'))
                torch.save({'epoch': epoch,
                            'encoder_model_state_dict': encoder_model.state_dict(),
                            'decoder_model_state_dict': decoder_model.state_dict(),
                            'encoder_optim_state_dict': encoder_optimizer.state_dict(),
                            'decoder_optim_state_dict': decoder_optimizer.state_dict(),
                            'train_loss': epoch_train_loss,
                            'val_loss': epoch_val_loss},
                           os.path.join(save_path, 'checkpoint', 'checkpoint_{}.pth'.format(epoch))
                           )
                
                # Save best model
                if not os.path.exists(os.path.join(save_path, 'best')):
                    os.makedirs(os.path.join(save_path, 'best'))
                if epoch_val_metric > best_metric:
                    best_metric = epoch_val_metric
                    best_metric_epoch = epoch
                    torch.save(encoder_model.state_dict(), os.path.join(save_path, 'best', 'best_encoder_model.pth'))
                    torch.save(decoder_model.state_dict(), os.path.join(save_path, 'best', 'best_decoder_model.pth'))

