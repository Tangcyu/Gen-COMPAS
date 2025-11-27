#!/usr/bin/env python3
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
from copy import deepcopy


@torch.compile
def train_one_epoch(epoch_index, tb_writer, train_loader, model_to_train, optimizer, loss_function, report_step, k_scale):
    print('EPOCH {} ({} batches):'.format(epoch_index + 1, len(train_loader)))
    running_loss = 0.0
    last_loss = 0.0
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader):
        # zero gradients for every batch
        optimizer.zero_grad(set_to_none=True)
        # compute the loss
        loss = loss_function(model_to_train, data,k_scale=k_scale)
        loss.backward()
        # adjust learning weights
        optimizer.step()
        # gather data and report
        running_loss += loss.item()
        if i % report_step == report_step - 1:
            last_loss = running_loss / report_step
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.0
    return last_loss


@torch.compile
def val_one_epoch(val_loader, model_to_train, loss_function, k_scale):
    running_loss = 0.0
    for data in val_loader:
        loss = loss_function(model_to_train, data, k_scale=k_scale)
        running_loss += loss.item()
    avg_loss = running_loss / len(val_loader)
    return avg_loss


def save_checkpoint(epoch, model, optimizer, writer_filename, best_epochs, best_model_state_dict, best_vloss, path):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'writer_filename': writer_filename,
                'best_epochs': best_epochs,
                'best_model_state_dict': best_model_state_dict,
                'best_vloss': best_vloss}, path)


def train_model(model_to_train, output_prefix, train_set, val_set, loss_function,
                epochs=1000, patience=20, batch_size_factor=0.6, learning_rate=1e-4, 
                old_checkpoint=None, epoch_metrics_callback=None, dataloader=None, load_old_model_only=False, k_scale=100.0):
    if dataloader is None:
        raise RuntimeError('Please provide a valid dataloader')
    # compute an appropriate batch size
    # see https://machine-learning.paperspace.com/wiki/epoch
    num_samples = len(train_set)
    # batch_size = int(np.sqrt(num_samples))
    # typically we have a training set larger than 1e7, so we should use a big batch size
    # if this is too small, then the batch cannot approximate the Koopman operator correctly
    batch_size = int(np.power(num_samples, batch_size_factor))
    print(f'Batch size: {batch_size}')
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    train_loader = dataloader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = dataloader(val_set, batch_size=len(val_set), shuffle=False)
    report_step = int(np.power(10, int(np.log10(len(train_loader)))))
    optimizer = torch.optim.Adam(model_to_train.parameters(), lr=learning_rate)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if old_checkpoint is not None:
        model_to_train.load_state_dict(old_checkpoint['model_state_dict'])
        if load_old_model_only is False:
            print('Will load the number of epochs and optimizer from previous training')
            epoch_number = old_checkpoint['epoch']
            optimizer.load_state_dict(old_checkpoint['optimizer_state_dict'])
            writer_filename = old_checkpoint['writer_filename']
            best_epochs = old_checkpoint['best_epochs']
            best_model_state_dict = deepcopy(old_checkpoint['best_model_state_dict'])
            best_vloss = old_checkpoint['best_vloss']
        else:
            print('Only the model is loaded from previous training')
    if old_checkpoint is None or load_old_model_only is True:
        epoch_number = 0
        writer_filename = f'{output_prefix}_trainer_{timestamp}'
        best_epochs = 0
        best_model_state_dict = deepcopy(model_to_train.state_dict())
        best_vloss = 1e7
    writer = SummaryWriter(writer_filename)
    checkpoint_filename = f'{output_prefix}_{timestamp}.checkpoint'
    while epoch_number < epochs:
        # Make sure gradient tracking is on, and do a pass over the data
        model_to_train.train(True)
        avg_loss = train_one_epoch(
            epoch_number, writer, train_loader, model_to_train, optimizer, loss_function, report_step, k_scale)
        # We don't need gradients on to do reporting
        model_to_train.train(False)
        # the loss of validation
        avg_vloss = val_one_epoch(val_loader, model_to_train, loss_function, k_scale)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        if epoch_metrics_callback is not None:
            results = epoch_metrics_callback(model_to_train, train_set, val_set)
            writer.add_scalars('Epoch metrics callback', results, epoch_number + 1)
        writer.flush()

        # Save intermediate models to plot an animation of the training
        #if epoch_number % 10==0:
        #if True:
        #    model_name = f'animation/{output_prefix}_best_model_{epoch_number}.pt'
        #    torch.jit.script(model_to_train).save(model_name)
        
        # Track the best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_model_state_dict = deepcopy(model_to_train.state_dict())
            model_name = f'{output_prefix}_best_model.pt'
            torch.jit.script(model_to_train).save(model_name)
            best_epochs = 0
        else:
            best_epochs += 1
            print(f'The best model has kept {best_epochs} epochs')
            if best_epochs >= patience:
                print('Early stopping!')
                break
        # save the checkpoint
        save_checkpoint(
            epoch_number + 1, model_to_train, optimizer, writer_filename,
            best_epochs, best_model_state_dict, best_vloss, checkpoint_filename)
        epoch_number += 1
    model_to_train.load_state_dict(best_model_state_dict)
    return model_to_train
