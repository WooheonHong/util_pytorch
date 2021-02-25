# %%
import time
import json
import random
from pathlib import Path
from datetime import datetime
from itertools import product
from collections import namedtuple
from collections import OrderedDict
from PIL import Image

from util_torch.torch_find_lr import find_lr
from util_torch.my_tensorboard import RunManager
from util_torch.my_tensorboard import RunBuilder
from util_torch.torch_datasplit import make_directory_split
from util_torch.torch_datasplit import DataSplit

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import librosa
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Timer:
    def __init__(self):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def get_time_hhmmss(self):
        end = time.time()
        m, s = divmod(end - self.start, 60)
        h, m = divmod(m, 60)
        time_str = f"{h:02.0f}:{m:02.0f}:{s:02.0f}"
        return time_str


def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)

        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)

        print(
            f"Epoch: {epoch}, Training Loss: {training_loss:.3f}, \
            Validation Loss: {valid_loss:.3f}, \
            accuracy = {num_correct / num_examples:.3f}"
        )


PATH_TO_MNIST = Path.cwd() / "MNIST"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dataset = torchvision.datasets.MNIST(
#     PATH_TO_MNIST,
#     train=True,
#     download=True,
#     transform=transforms.Compose(
#         [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
#     ),
# )


def precompute_spectrogram(path, dpi=50):
    files = Path(path).glob("*.wav")
    for filename in files:
        audio_tensor, sample_rate = librosa.load(filename, sr=None)
        spectrogram = librosa.feature.melspectrogram(audio_tensor, sr=sample_rate)
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        librosa.display.specshow(
            log_spectrogram, sr=sample_rate, x_axis="time", y_axis="mel"
        )
        plt.gcf().savefig(f"{filename.parent}/{dpi}_{filename.name}.png", dpi=dpi)


class FrequencyMask(object):
    """
      Example:
        >>> transforms.Compose([
            transforms.ToTensor(),
            FrequencyMask(max_width=10, use_maen=False),
        ])
    """

    def __init__(self, max_width, use_mean=True):
        self.max_width = max_width
        self.use_mean = use_mean

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) where 
            the frequency mask is to be applied.

        Returns:
            Tensor: Transformed image with Frequency Mask.
        """
        start = random.randrange(0, tensor.shape[2])
        end = start + random.randrange(1, self.max_width)
        if self.use_mean:
            tensor[:, start:end, :] = tensor.mean()
        else:
            tensor[:, start:end, :] = 0
        return tensor

    def __repr__(self):
        # format_string = self.__class__.__name__ + "(max_width="
        # format_string += str(self.max_width) + ")"
        # format_string += 'use_mean=' + (str(self.use_mean) + ')')

        # return format_string
        return f"{self.__class__.name}(max_width={str(self.max_width)}, use_mean={str(self.use_mean)})"


class TimeMask(object):
    """
        Example:
        >>> transforms.Compose([
                transforms.ToTensor(),
                TimeMask(max_widht=10, use_mean=False),
        ])
    """

    def __init__(self, max_width, use_mean=True):
        self.max_width = max_width
        self.use_mean = use_mean

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W)
            where the time mask is to be applied.
        
        Returns:
            Tensor: Transformed image iwth Time Mask.

        """
        start = random.randrange(0, tensor.shape[1])
        end = start + random.randrange(0, self.max_width)
        if self.use_mean:
            tensor[:, :, start:end] = tensor.mean()
        else:
            tensor[:, :, start:end] = 0
        return tensor

    def __repr__(self):

        return f"{self.__class__.__name__}'(max_width='{str(self.max_width)})'use_mean='{str(self.use_mean)})"


class PrecomputedTransformESC50(Dataset):
    def __init__(
        self,
        path,
        max_freqmask_width,
        max_timemask_width,
        img_transforms=None,
        use_mean=True,
        dpi=50,
    ):

        files = Path(path).glob(f"{dpi}*.wav.png")
        self.items = [
            (f, int(f.name.split("-")[-1].replace(".wav.png", ""))) for f in files
        ]
        self.length = len(self.items)
        self.max_freqmask_width = max_freqmask_width
        self.max_timemask_width = max_timemask_width
        self.use_mean = use_mean
        if img_transforms == None:
            self.img_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                    transforms.RandomApply(
                        [FrequencyMask(self.max_freqmask_width, self.use_mean)], p=0.5
                    ),
                    transforms.RandomApply(
                        [TimeMask(self.max_timemask_width, self.use_mean)], p=0.5
                    ),
                ]
            )
        else:
            self.img_transforms = img_transforms

    def __getitem__(self, index):
        filename, label = self.items[index]
        img = Image.open(filename).convert("RGB")
        return (self.img_transforms(img), label)

    def __len__(self):
        return self.length

# %%
timer = Timer()
precompute_spectrogram(Path.cwd() / "ESC-50" / "audio")
PATH_TO_ESC50 = Path.cwd() / "ESC-50" / "audio"
print(f"precompute spectrogram: {timer.get_time_hhmmss()}")

# %%
# trian, val, test 디렉토리 생성
timer.restart()
split = make_directory_split(PATH_TO_ESC50)
# png 생성
PATH_ESC50_TRAIN = PATH_TO_ESC50 / "train"
PATH_ESC50_VAL = PATH_TO_ESC50 / "validation"
print(f"make_directory_split: {timer.get_time_hhmmss()}")

# %%
esc50pre_train = PrecomputedTransformESC50(
    PATH_ESC50_TRAIN, max_freqmask_width=10, max_timemask_width=10,
)
timer.restart()
esc50pre_val = PrecomputedTransformESC50(
    PATH_ESC50_VAL, max_freqmask_width=10, max_timemask_width=10
)

esc50_train_loader = torch.utils.data.DataLoader(esc50pre_train, 32, shuffle=True)
esc50_val_loader = torch.utils.data.DataLoader(esc50pre_val, 32, shuffle=True)
# %%
# find initial leraning rate 
model = torchvision.models.resnet50(True)
for name, param in model.named_parameters():
    if "bn" not in name:
        param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 500), nn.ReLU(), nn.Dropout(), nn.Linear(500, 50),
)
model.to(device)
torch.save(model.state_dict(), "model_resnet50.pth")
loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
lrs, losses = find_lr(model, loss_fn, optimizer, esc50_train_loader, device=device)
plt.plot(lrs, losses)
plt.xscale("log")
plt.xlabel("Learning rate")
plt.ylabel("Loss")
plt.show()

# %%
model.load_state_dict(torch.load("model_resnet50.pth"))
found_lr = 0.001
params = OrderedDict(
    lr_tuning=[50, 100, 200], batch_size=[64], shuffle=[True], device=["cuda"],
)


def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print("Inside " + self.__class__.__name__ + " forward")
    print("")
    print("input: ", type(input))
    print("input[0]: ", type(input[0]))
    print("output: ", type(output))
    print("")
    print("input size:", input[0].size())
    print("output size:", output.data.size())
    print("output norm:", output.data.norm())


# train for a few epochs that update only the classifier
optimizer = optim.Adam(model.parameters(), lr=found_lr)
train(
    model,
    optimizer,
    loss_fn,
    esc50_train_loader,
    esc50_val_loader,
    epochs=5,
    device="cuda",
)

epochs = 20
manager = RunManager(log_dir="test")

for run in RunBuilder.get_runs(params):
    model.to(run.device)
    esc50_train_loader = torch.utils.data.DataLoader(
        esc50pre_train, run.batch_size, shuffle=run.shuffle
    )
    esc50_val_loader = torch.utils.data.DataLoader(
        esc50pre_val, run.batch_size, shuffle=run.shuffle
    )

    optimizer = optim.Adam(
        [
            {"params": model.conv1.parameters()},
            {"params": model.bn1.parameters()},
            {"params": model.relu.parameters()},
            {"params": model.maxpool.parameters()},
            {"params": model.layer1.parameters(), "lr": found_lr / run.lr_tuning},
            {"params": model.layer2.parameters(), "lr": found_lr / run.lr_tuning},
            {"params": model.layer3.parameters(), "lr": found_lr / run.lr_tuning},
            {"params": model.layer4.parameters(), "lr": found_lr / run.lr_tuning},
            {"params": model.avgpool.parameters(), "lr": found_lr / run.lr_tuning},
            {"params": model.fc.parameters(), "lr": found_lr / (100 * run.lr_tuning),},
        ],
        lr=found_lr,
    )
    for param in model.parameters():
        param.requires_grad = True

    manager.begin_run(run, model, esc50_train_loader, esc50_val_loader)

    for epoch in range(epochs):
        manager.begin_epoch()
        model.train()
        for batch in esc50_train_loader:

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
            for batch in esc50_val_loader:
                images = batch[0].to(run.device)
                labels = batch[1].to(run.device)
                preds = model(images)  # Pass Batch
                loss = F.cross_entropy(preds, labels)  # Calculate Loss

                manager.track_loss(loss, batch, is_train=False)
                manager.track_num_correct(preds, labels)

        manager.end_epoch()
    manager.end_run()
manager.save("results")
