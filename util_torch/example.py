import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import time
import json
from pathlib import Path
from datetime import datetime

from util_torch.torch_find_lr import find_lr
from util_torch.my_tensorboard import RunManager
from util_torch.my_tensorboard import RunBuilder

from itertools import product
from collections import namedtuple
from collections import OrderedDict

from util_torch.torch_datasplit import DataSplit
import matplotlib.pyplot as plt


PATH_TO_MNIST = Path.cwd() / "MNIST"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = torchvision.datasets.MNIST(
    PATH_TO_MNIST,
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    ),
)

model = torchvision.models.resnet50(True)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.to(device)
loss_fn = nn.CrossEntropyLoss()
split = DataSplit(dataset, shuffle=True)
train_loader, val_loader, test_loader = split.get_split(batch_size=32, num_workers=4)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
logs, losses = find_lr(model, loss_fn, optimizer, train_loader, device=device)
plt.plot(logs, losses)

found_lr = [0.0003, 0.0004, 0.0005]
params = OrderedDict(lr=found_lr, batch_size=[64], shuffle=[True], device=["cuda"],)


def train(model, params, epochs=20):
    manager = RunManager()

    for run in RunBuilder.get_runs(params):
        model.to(run.device)
        split = DataSplit(dataset, shuffle=run.shuffle)
        train_loader, val_loader, test_loader = split.get_split(
            batch_size=run.batch_size, num_workers=4
        )
        optimizer = optim.Adam(
            [
                {"params": model.layer3.parameters(), "lr": run.lr / 3},
                {"params": model.layer4.parameters(), "lr": run.lr / 9},
                {"params": model.avgpool.parameters(), "lr": run.lr / 9},
                {"params": model.fc.parameters(), "lr": run.lr / 100},
            ],
            lr=run.lr,
        )
        unfreeze_layers = [model.layer3, model.layer4, model.avgpool, model.fc]
        for layer in unfreeze_layers:
            for param in layer.parameters():
                param.requires_grad = True

        manager.begin_run(run, model, train_loader, val_loader)

        for epoch in range(epochs):
            manager.begin_epoch()
            model.train()
            for batch in train_loader:

                images = batch[0].to(run.device)
                labels = batch[1].to(run.device)
                preds = model(images)  # Pass Batch
                loss = F.cross_entropy(preds, labels)  # Calculate Loss
                optimizer.zero_grad()  # Zero Gradients
                loss.backward()  # Calculate Gradients
                optimizer.step()  # Update Weights

                manager.track_loss(loss, batch, is_train=True)

            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    images = batch[0].to(run.device)
                    labels = batch[1].to(run.device)
                    preds = model(images)  # Pass Batch
                    loss = F.cross_entropy(preds, labels)  # Calculate Loss

                    manager.track_loss(loss, batch, is_train=False)
                    manager.track_num_correct(preds, labels)

            manager.end_epoch()
        manager.end_run()
    manager.save("results")


train(model, params, epochs=20)

