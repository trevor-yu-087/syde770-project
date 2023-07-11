import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import model.hyperparameters as hp

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

        for train_step, train_data in enumerate(train_loader):
            train_source = train_data['encoder_inputs'].to(device)
            train_target = train_data['decoder_inputs'].to(device)

            # Zero optimizers
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # Forward pass
            decoder_output = torch.zeros(hp.BATCH_SIZE, 512, 7).to(device)
            train_target_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(train_target, batch_first=True)
            train_target_unpacked.to(device)
            # start = train_target_unpacked[:, 0, :].unsqueeze(1).to(device)
            teacher_force = True if random.random() < teacher_force_ratio else False

            encoder_hidden, encoder_cell = encoder_model(train_source)
            # print(encoder_hidden.shape)
            encoder_cell = torch.zeros(encoder_cell.shape).to(device)
            
            if train_step == 0:
                decoder_output, decoder_hidden, decoder_cell = decoder_model(train_target, encoder_hidden, encoder_cell)
                # print(f'Decoder Output: {decoder_output.shape}\t Decoder Hidden: {decoder_hidden.shape}\t Decoder Cell: {decoder_cell.shape}')
            elif train_step !=0 and teacher_force == True:
                decoder_output, decoder_hidden, decoder_cell = decoder_model(train_target, encoder_hidden, encoder_cell)
            elif train_step != 0 and teacher_force == False:
                for i in range(0, train_target_unpacked.shape[1]): # cycle through all elements of sequence
                    start = train_target_unpacked[:, i, :].unsqueeze(1).to(device)
                    start = [start[i] for i in range(start.shape[0])]
                    start = torch.nn.utils.rnn.pack_sequence(start)
                    decoder_output[:, i, :], decoder_hidden, decoder_cell = decoder_model(start, encoder_hidden, encoder_cell)
                    encoder_hidden = decoder_hidden
                    encoder_cell = decoder_cell

            train_loss = loss_fn(decoder_output, train_target_unpacked)

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
            train_metric = metric_loss_fn(decoder_output, train_target_unpacked)
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
                    val_source = val_data['encoder_inputs'].to(device)
                    val_target = val_data['decoder_inputs'].to(device)
                    val_target_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(val_target, batch_first=True)
                    val_target_unpacked.to(device)

                    # Run validation model
                    val_encoder_hidden, val_encoder_cell = encoder_model(val_source)
                    val_encoder_cell = torch.zeros(val_encoder_cell.shape).to(device)

                    val_decoder_output, val_decoder_hidden, val_decoder_cell = decoder_model(val_target, val_encoder_hidden, val_encoder_cell)

                    val_loss = loss_fn(val_decoder_output, val_target_unpacked)

                    # Val loss
                    epoch_val_loss += val_loss.item()

                    # Val metric loss
                    val_metric = metric_loss_fn(val_decoder_output, val_target_unpacked)
                    epoch_val_metric += val_metric

                # Average validation losses for tensorboard
                epoch_val_loss /= (val_step+1)
                writer.add_scalar('Validation MSE per Epoch', epoch_val_loss, epoch)
                val_loss_values.append(epoch_val_loss)
                epoch_val_metric /= (val_step+1)
                writer.add_scalar('Validation MAE per Epoch', epoch_val_metric, epoch)
                val_metric_values.append(epoch_val_metric)


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
        val_interval=1,
        checkpoint=None,
        batch_size=1
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
            train_source = train_data['encoder_inputs'].to(device)
            source_padding = train_data['encoder_padding_mask'].to(device)
            train_target = train_data['decoder_inputs'].to(device)
            target_padding = train_data['decoder_padding_mask'].to(device)
            target_lookahead = train_data['decoder_lookahead_mask'].to(device)

            # Zero optimizers
            transformer_optimizer.zero_grad()

            # Forward pass
            transformer_output = torch.zeros(batch_size, 512, 7).to(device)
            train_target.to(device)
            #src_start = train_source[:, 0, :].unsqueeze(1).to(device)
            start = train_target[:, 0, :].unsqueeze(1).to(device)
            teacher_force = True if random.random() < teacher_force_ratio else False


            if train_step == 0:
                transformer_output = transformer_model(src=train_source, tgt=train_target, src_padding=source_padding, 
                                                        tgt_padding=target_padding, tgt_lookahead=target_lookahead)
                

                # print(f'Decoder Output: {decoder_output.shape}\t Decoder Hidden: {decoder_hidden.shape}\t Decoder Cell: {decoder_cell.shape}')
            elif train_step !=0 and teacher_force == True:
                transformer_output = transformer_model(src=train_source, tgt=train_target, src_padding=source_padding, 
                                                        tgt_padding=target_padding, tgt_lookahead=target_lookahead)

            elif train_step != 0 and teacher_force == False:
                for i in range(1, 512):
                    transformer_output[:, i, :] = transformer_model(src=train_source, tgt=start)
                    start = train_target[:, i, :].unsqueeze(1)


            train_loss = loss_fn(transformer_output, train_target)

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
                    val_source = val_data['encoder_inputs'].to(device)
                    val_target = val_data['decoder_inputs'].to(device)
                    
                    # Run validation model
                    val_transformer_output = transformer_model(src=val_source, tgt=val_target)

                    val_loss = loss_fn(val_transformer_output, val_target)

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
