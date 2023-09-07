import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import model.hyperparameters as hp
from utils.visualize import pred_vs_error

def CNN_train_fn(
        train_loader,
        val_loader,
        model,
        optimizer,
        loss_fn,
        metric_loss_fn,
        num_epoch,
        device,
        save_path,
        writer,
        enable_checkpoint=False,
        checkpoint=None,
        val_interval=1,
):
    best_metric = 1e4
    val_loss_values = []
    val_metric_values = []

    train_std, train_mean = get_stats(train_loader)
    val_std, val_mean = get_stats(val_loader)

    for epoch in range(num_epoch):
        print(f'===== Epoch: {epoch} =====')
        epoch_train_loss = 0
        epoch_train_metric = 0
        model.train()
        preds, targets = [], []

        for train_step, train_data in enumerate(train_loader):
            train_source = train_data['inputs'].to(device)
            train_target = train_data['targets']
            # zero-score normalize velocity targets
            for i in range(3):
                train_target[:,i] = (train_target[:,i] - train_mean[i]) / train_std[i]
                # train_target[:,i] /= train_std[i]
            train_target = train_target.to(device)

            # Zero optimizers
            optimizer.zero_grad()

            # Forward pass
            pred = model(train_source) # error in model
            preds.append(pred.cpu().detach().numpy())
            targets.append(train_target.cpu().detach().numpy())

            train_loss = loss_fn(pred, train_target)

            if torch.isnan(train_loss):
                print(f'Epoch: {epoch} \t Step: {train_step}')
                raise ValueError('Train loss returns NAN value')

            # Backwards
            train_loss.backward()

            # # gradient clipping
            # nn.utils.clip_grad_norm_(encoder_model.parameters(), 5)
            # nn.utils.clip_grad_norm_(decoder_model.parameters(), 5)

            # Update optimizers
            optimizer.step()

            # Train loss
            epoch_train_loss += train_loss.item()

            # Train metric loss
            train_metric = metric_loss_fn(pred, train_target)
            epoch_train_metric += train_metric

        # Average losses for tensorboard
        epoch_train_loss /= (train_step+1)
        writer.add_scalar('Training MSE per Epoch', epoch_train_loss, epoch)
        epoch_train_metric /= (train_step+1)
        writer.add_scalar('Training MAE per Epoch', epoch_train_metric, epoch)
        
        # if epoch == 15 or epoch == 40:
        #     pred_vs_error(epoch, preds, targets)
        

        if (epoch+1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                epoch_val_loss = 0
                epoch_val_metric = 0

                for val_step, val_data in enumerate(val_loader):
                    val_source = val_data['inputs'].to(device)
                    val_target = val_data['targets']
                    # zero-score normalize velocity targets
                    for i in range(3):
                        val_target[:,i] = (val_target[:,i] - val_mean[i]) / val_std[i]
                    val_target = val_target.to(device)

                    # Run validation model
                    val_pred = model(val_source)

                    val_loss = loss_fn(val_pred, val_target)

                    if torch.isnan(val_loss):
                        print(f'Epoch: {epoch} \t Step: {val_step}')
                        # raise ValueError('Val loss returns NAN value')

                    # Val loss
                    epoch_val_loss += val_loss.item()

                    # Val metric loss
                    val_metric = metric_loss_fn(val_pred, val_target)
                    epoch_val_metric += val_metric

                # Average validation losses for tensorboard
                epoch_val_loss /= (val_step+1)
                writer.add_scalar('Validation MSE per Epoch', epoch_val_loss, epoch)
                val_loss_values.append(epoch_val_loss)
                epoch_val_metric /= (val_step+1)
                writer.add_scalar('Validation MAE per Epoch', epoch_val_metric, epoch)
                val_metric_values.append(epoch_val_metric)
                
                print(f"Epoch {epoch} MSE Loss: {epoch_val_loss}")
                print(f"Epoch {epoch} MAE: {epoch_val_metric}")

                # Save checkpoint
                if enable_checkpoint:
                    if not os.path.exists(os.path.join(save_path, 'checkpoint')):
                        os.makedirs(os.path.join(save_path, 'checkpoint'))
                    torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optim_state_dict': optimizer.state_dict(),
                                'train_loss': epoch_train_loss,
                                'val_loss': epoch_val_loss},
                            os.path.join(save_path, 'checkpoint', 'checkpoint_{}.pth'.format(epoch))
                            )
                    
                    # Save best model
                    if not os.path.exists(os.path.join(save_path, 'best')):
                        os.makedirs(os.path.join(save_path, 'best'))
                    if epoch_val_metric < best_metric:
                        best_metric = epoch_val_metric
                        best_metric_epoch = epoch
                        torch.save(model.state_dict(), os.path.join(save_path, 'best', 'best_model.pth'))

    writer.close()
    return val_loss_values

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
        teacher_force_ratio,
        dynanmic_tf,
        enable_checkpoint=False,
        checkpoint=None,
        val_interval=1,
):
    best_metric = 1e4
    val_loss_values = []
    val_metric_values = []

    for epoch in range(num_epoch):
        print(f'===== Epoch: {epoch} =====')
        epoch_train_loss = 0
        epoch_train_metric = 0
        encoder_model.train()
        decoder_model.train()
        print(f'Teacher Force Ratio:{teacher_force_ratio}')
        
        # if epoch % 1 == 0 and epoch > 4:
        #         teacher_force_ratio *= 0.8 
        #         print(f'Teacher Force Ratio:{teacher_force_ratio}')

        for train_step, train_data in enumerate(train_loader):
            train_enc_source = train_data['encoder_inputs'].to(device)
            train_dec_source = train_data['decoder_inputs'].to(device)
            train_target = train_data['targets'].to(device)
            # # unpack train_dec_source for AR training
            # train_dec_source_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(train_dec_source, batch_first=True)
            # train_dec_source_unpacked.to(device)
            # # unpack train_target for loss functions
            # train_target_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(train_target, batch_first=True)
            # train_target_unpacked.to(device)

            # Zero optimizers
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # Forward pass
            decoder_output = torch.zeros((train_target.shape)).to(device)
            # teacher_force = True if random.random() < teacher_force_ratio else False

            encoder_hidden, encoder_cell = encoder_model(train_enc_source)

            # initialize seed for decoder
            decoder_input = train_dec_source[:,0,:].unsqueeze(1)

            if random.random() < teacher_force_ratio: # teacher force targets
                output, hidden, cell = decoder_model(decoder_input, encoder_hidden, encoder_cell)
                decoder_output[:,0,:] = output
                for i in range(1, train_target.shape[1]):
                    decoder_input = train_dec_source[:,i,:].unsqueeze(1)
                    output, hidden, cell = decoder_model(decoder_input, encoder_hidden, encoder_cell)
                    decoder_output[:,i,:] = output
            else: # auto-regressive generation
                output, hidden, cell = decoder_model(decoder_input, encoder_hidden, encoder_cell)
                decoder_output[:,0,:] = output
                for i in range (1, train_target.shape[1]):
                    decoder_input = output.unsqueeze(1)
                    output, hidden, cell = decoder_model(decoder_input, encoder_hidden, encoder_cell)
                    decoder_output[:,i,:] = output.squeeze()

            if dynanmic_tf and teacher_force_ratio > 0.01:
                teacher_force_ratio -= 0.01
            else:
                teacher_force_ratio = 0.0
                


            # # print(encoder_hidden.shape)
            # # encoder_cell = torch.zeros(encoder_cell.shape).to(device)
            # # if epoch < min_teacher_force:
            # #      decoder_output, decoder_hidden, decoder_cell = decoder_model(train_dec_source, encoder_hidden, encoder_cell)
            # # else:
            # if train_step == 0:
            #     decoder_output, decoder_hidden, decoder_cell = decoder_model(train_dec_source, encoder_hidden, encoder_cell)
            #     # print(f'Decoder Output: {decoder_output.shape}\t Decoder Hidden: {decoder_hidden.shape}\t Decoder Cell: {decoder_cell.shape}')
            # elif train_step !=0 and teacher_force == True:
            #     decoder_output, decoder_hidden, decoder_cell = decoder_model(train_dec_source, encoder_hidden, encoder_cell)
            # # autoregressive training
            # elif train_step != 0 and teacher_force == False: 
            #     start = train_dec_source_unpacked[:,0,:].unsqueeze(1).to(device)
            #     # start = [start[i,:,:] for i in range(start.shape[0])]
            #     # start = torch.nn.utils.rnn.pack_sequence(start)
            #     output, (hidden, cell) = decoder_model.LSTM(start, (encoder_hidden, encoder_cell))
            #     # forward_output, backward_output = torch.split(output, split_size_or_sections=32, dim=2)
            #     output = decoder_model.fc(output)
            #     decoder_output[:,0,:] = output.squeeze()

            #     for i in range(1, train_dec_source_unpacked.shape[1]): # cycle through all elements of sequence
            #         output, (hidden, cell) = decoder_model.LSTM(output,( hidden, cell))
            #         # forward_output, backward_output = torch.split(output, split_size_or_sections=32, dim=2)
            #         output = decoder_model.fc(output)
            #         decoder_output[:,i,:] = output.squeeze()

            train_loss = loss_fn(decoder_output, train_target)

            if torch.isnan(train_loss):
                print(f'Epoch: {epoch} \t Step: {train_step}')
                raise ValueError('Train loss returns NAN value')

            # Backwards
            train_loss.backward()

            # # gradient clipping
            # nn.utils.clip_grad_norm_(encoder_model.parameters(), 5)
            # nn.utils.clip_grad_norm_(decoder_model.parameters(), 5)

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
        

        if (epoch+1) % val_interval == 0:
            encoder_model.eval()
            decoder_model.eval()
            with torch.no_grad():
                epoch_val_loss = 0
                epoch_val_metric = 0

                for val_step, val_data in enumerate(val_loader):
                    val_enc_source = val_data['encoder_inputs'].to(device)
                    val_dec_source = val_data['decoder_inputs'].to(device)
                    val_target = val_data['targets'].to(device)
                    # # unpack val_dec_source for auto-regressive generation
                    # val_dec_source_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(val_dec_source, batch_first=True)
                    # val_dec_source_unpacked.to(device)
                    # # unpack val_target for loss functions
                    # val_target_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(val_target, batch_first=True)
                    # val_target_unpacked.to(device)

                    # Run validation model
                    val_encoder_hidden, val_encoder_cell = encoder_model(val_enc_source)
                    # val_encoder_cell = torch.zeros(val_encoder_cell.shape).to(device)

                    # auto-regressive decoder output generation
                    val_dec_output = torch.zeros((val_target.shape)).to(device)

                    val_dec_input =  val_dec_source[:,0,:].unsqueeze(1)
                    output, hidden, cell = decoder_model(val_dec_input, val_encoder_hidden, val_encoder_cell)
                    val_dec_output[:,0,:] = output

                    for i in range(1, val_target.shape[1]):
                        val_dec_input = output.unsqueeze(1)
                        output, hidden, cell = decoder_model(val_dec_input, val_encoder_hidden, val_encoder_cell)
                        val_dec_output[:,i,:] = output

                    # decoder_seed = val_dec_source[:,0,:].unsqueeze(1)
                    # output, (hidden, cell) = decoder_model.LSTM(decoder_seed,(val_encoder_hidden, val_encoder_cell))
                    # # forward_output, backward_output = torch.split(output, split_size_or_sections=32, dim=2)
                    # output = decoder_model.fc(output)
                    # val_dec_output[:,0,:] = output.squeeze()

                    # for i in range(1, val_dec_source.shape[1]):
                    #     output, (hidden, cell) = decoder_model.LSTM(output, (hidden, cell))
                    #     # forward_output, backward_output = torch.split(output, split_size_or_sections=32, dim=2)
                    #     output = decoder_model.fc(output)
                    #     val_dec_output[:,i,:] = output.squeeze()

                    val_loss = loss_fn(val_dec_output, val_target)

                    if torch.isnan(val_loss):
                        print(f'Epoch: {epoch} \t Step: {val_step}')
                        # raise ValueError('Val loss returns NAN value')

                    # Val loss
                    epoch_val_loss += val_loss.item()

                    # Val metric loss
                    val_metric = metric_loss_fn(val_dec_output, val_target)
                    epoch_val_metric += val_metric

                # Average validation losses for tensorboard
                epoch_val_loss /= (val_step+1)
                writer.add_scalar('Validation MSE per Epoch', epoch_val_loss, epoch)
                val_loss_values.append(epoch_val_loss)
                epoch_val_metric /= (val_step+1)
                writer.add_scalar('Validation MAE per Epoch', epoch_val_metric, epoch)
                val_metric_values.append(epoch_val_metric)
                
                print(f"Epoch {epoch} MSE Loss: {epoch_val_loss}")
                print(f"Epoch {epoch} MAE: {epoch_val_metric}")

                # Save checkpoint
                if enable_checkpoint:
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
                    if epoch_val_metric < best_metric:
                        best_metric = epoch_val_metric
                        best_metric_epoch = epoch
                        torch.save(encoder_model.state_dict(), os.path.join(save_path, 'best', 'best_encoder_model.pth'))
                        torch.save(decoder_model.state_dict(), os.path.join(save_path, 'best', 'best_decoder_model.pth'))

    writer.close()
    return val_loss_values


def Transformer_train_fn(
        train_loader,
        val_loader,
        transformer_model,
        transformer_optimizer,
        loss_fn,
        metric_loss_fn,
        num_epoch,
        device,
        save_path,
        writer,
        teacher_force_ratio=1,
        enable_checkpoint=False,
        checkpoint=None,
        batch_size=1,
        val_interval=1,
):
    best_metric = 1e4
    val_loss_values = []
    val_metric_values = []

    for epoch in range(num_epoch):
        print(f'===== Epoch: {epoch} =====')
        epoch_train_loss = 0
        epoch_train_metric = 0
        transformer_model.train()

        for train_step, train_data in enumerate(train_loader):
            train_enc_source = train_data['encoder_inputs'].to(device)
            source_padding = train_data['encoder_padding_mask'].to(device)
            train_dec_source = train_data['decoder_inputs'].to(device)
            target_padding = train_data['decoder_padding_mask'].to(device)
            target_lookahead = train_data['decoder_lookahead_mask'].to(device)
            train_target = train_data['targets'].to(device)

            # Zero optimizers
            transformer_optimizer.zero_grad()

            # Forward pass
            # transformer_output = torch.zeros((train_target.shape)).to(device)
            teacher_force = True if random.random() < teacher_force_ratio else False

            if train_step == 0:
                transformer_output = transformer_model(
                    src=train_enc_source, 
                    tgt=train_dec_source, 
                    src_padding=source_padding, 
                    tgt_padding=target_padding, 
                    tgt_lookahead=target_lookahead
                )
                
            elif train_step !=0 and teacher_force == True:
                transformer_output = transformer_model(
                    src=train_enc_source, 
                    tgt=train_dec_source, 
                    src_padding=source_padding, 
                    tgt_padding=target_padding, 
                    tgt_lookahead=target_lookahead
                )
            # autoregressive training
            elif train_step != 0 and teacher_force == False:
                start = train_dec_source[:,0,:].unsqueeze(1)
                output = transformer_model(
                    src=train_enc_source, 
                    tgt=start, 
                )
                transformer_output = torch.cat([start, output.unsqueeze(1)], dim=1)

                for i in range(1, train_dec_source.shape[1]):
                    output = transformer_model(src=train_enc_source, tgt=transformer_output)
                    transformer_output = torch.cat([transformer_output, output[:,-1,:].unsqueeze(1)], dim=1)
                transformer_output = transformer_output[:,1:,:]

            train_loss = loss_fn(transformer_output, train_target)

            if torch.isnan(train_loss):
                raise ValueError(f'Train loss returns NAN value \nEpoch: {epoch} \t Step: {train_step}')

            # Backwards
            train_loss.backward()

            # Update optimizers
            transformer_optimizer.step()

            # Train loss
            epoch_train_loss += train_loss.item()

            # Train metric loss
            train_metric = metric_loss_fn(transformer_output, train_target)
            epoch_train_metric += train_metric

        # Average losses for tensorboard
        epoch_train_loss /= (train_step+1)
        writer.add_scalar('Training MSE per Epoch', epoch_train_loss, epoch)
        epoch_train_metric /= (train_step+1)
        writer.add_scalar('Training MAE per Epoch', epoch_train_metric, epoch)
        

        if (epoch+1) % val_interval == 0:
            transformer_model.eval()
            with torch.no_grad():
                epoch_val_loss = 0
                epoch_val_metric = 0

                for val_step, val_data in enumerate(val_loader):
                    val_enc_source = val_data['encoder_inputs'].to(device)
                    val_dec_source = val_data['decoder_inputs'].to(device)
                    val_target = val_data['targets'].to(device)
                    
                    # Run validation model
                    # val_transformer_output =  torch.zeros((val_target.shape))
                    start = val_dec_source[:,0,:].unsqueeze(1)
                    output = transformer_model(src=val_enc_source, tgt=start)
                    val_transformer_output = torch.cat([start, output.unsqueeze(1)], dim=1)

                    for i in range(1, val_target.shape[1]):
                        output = transformer_model(src=val_enc_source, tgt=val_transformer_output)
                        val_transformer_output = torch.cat([val_transformer_output, output[:,-1,:].unsqueeze(1)], dim=1)
                    val_transformer_output = val_transformer_output[:,1:,:]

                    val_loss = loss_fn(val_transformer_output, val_target) # perform loss calc without seed position

                    if torch.isnan(val_loss):
                        print(f'Loss NAN - Epoch: {epoch} \t Step: {val_step}')
                        # raise ValueError('Val loss returns NAN value')

                    # Val loss
                    epoch_val_loss += val_loss.item()

                    # Val metric loss
                    val_metric = metric_loss_fn(val_transformer_output, val_target)
                    epoch_val_metric += val_metric

                # Average validation losses for tensorboard
                epoch_val_loss /= (val_step+1)
                writer.add_scalar('Validation MSE per Epoch', epoch_val_loss, epoch)
                val_loss_values.append(epoch_val_loss)
                epoch_val_metric /= (val_step+1)
                writer.add_scalar('Validation MAE per Epoch', epoch_val_metric, epoch)
                val_metric_values.append(epoch_val_metric)
                
                print(f"Epoch {epoch} MSE Loss: {epoch_val_loss}")
                print(f"Epoch {epoch} MAE: {epoch_val_metric}")


                # Save checkpoint
                if enable_checkpoint:
                    if not os.path.exists(os.path.join(save_path, 'checkpoint')):
                        os.makedirs(os.path.join(save_path, 'checkpoint'))
                    torch.save({'epoch': epoch,
                                'transformer_model_state_dict': transformer_model.state_dict(),
                                'transformer_optim_state_dict': transformer_optimizer.state_dict(),
                                'train_loss': epoch_train_loss,
                                'val_loss': epoch_val_loss},
                            os.path.join(save_path, 'checkpoint', 'checkpoint_{}.pth'.format(epoch))
                            )
                    
                    # Save best model
                    if not os.path.exists(os.path.join(save_path, 'best')):
                        os.makedirs(os.path.join(save_path, 'best'))
                    if epoch_val_metric < best_metric:
                        best_metric = epoch_val_metric
                        best_metric_epoch = epoch
                        torch.save(transformer_model.state_dict(), os.path.join(save_path, 'best', 'best_transformer_model.pth'))

    writer.close()
    return val_loss_values


def get_stats(
        loader
):
    import numpy as np
    x, y, z = [], [], []
    for step, data in enumerate(loader):
        targets = data['targets']
        x.append(targets[:, 0].numpy())
        y.append(targets[:, 1].numpy())
        z.append(targets[:, 2].numpy())
                
    x = np.concatenate(np.array(x), axis=0)
    y = np.concatenate(np.array(y), axis=0)
    z = np.concatenate(np.array(z), axis=0)
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
    return velocity_std, velocity_mean