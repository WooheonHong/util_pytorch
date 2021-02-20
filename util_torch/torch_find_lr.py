"""
Over the course of an epoch, start out with a small learning rate 
and increase to a higher learning rate over each mini-batch, 
resulting in a high rate at the end of the epoch. Calculate the loss for each rate 
and then, looking at a plot, pick the learning rate that gives the greatest decline

------------------------------------usage----------------------------------------------
(lrs, losses) = find_lr(transfer_model, torch.nn.CrossEntropyLoss(), 
                        optimizer, train_data_loader,device=device)
plt.plot(lrs, losses)

plt.xscale("log")
plt.xlabel("Learning rate")
plt.ylabel("Loss")
plt.show()
"""

import numpy as np
import random
import time

import torch
import torchaudio
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def find_lr(
    model,
    loss_fn,
    optimizer,
    train_loader,
    init_value=1e-8,
    final_value=10.0,
    device="cpu",
):
    number_in_epoch = len(train_loader) - 1
    update_step = (final_value / init_value) ** (1 / number_in_epoch)
    lr = init_value
    optimizer.param_groups[0]["lr"] = lr
    best_loss = 0.0
    batch_num = 0
    losses = []
    log_lrs = []
    for data in train_loader:
        batch_num += 1
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # Crash out if loss explodes

        if batch_num > 1 and loss > 4 * best_loss:
            if len(log_lrs) > 20:
                return log_lrs[10:-5], losses[10:-5]
            else:
                return log_lrs, losses

        # Record the best loss

        if loss < best_loss or batch_num == 1:
            best_loss = loss

        # Store the values
        losses.append(loss.item())
        log_lrs.append((lr))

        # Do the backward pass and optimize

        loss.backward()
        optimizer.step()

        # Update the lr for the next step and store

        lr *= update_step
        optimizer.param_groups[0]["lr"] = lr
    if len(log_lrs) > 20:
        return log_lrs[10:-5], losses[10:-5]
    else:
        return log_lrs, losses
