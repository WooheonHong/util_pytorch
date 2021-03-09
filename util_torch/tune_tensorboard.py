import time
import json
from pathlib import Path
from itertools import product
from collections import namedtuple
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, clear_output
import pandas as pd


class RunBuilder:
    @staticmethod
    def get_runs(params):

        Run = namedtuple("Run", params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


class RunManager:
    def __init__(self, log_dir=None, filename_suffix="", is_clf=True, is_image=False):
        self.log_dir = log_dir
        self.filename_suffix = filename_suffix
        self.is_clf = is_clf
        self.is_image = is_image

        self.epoch_count = 0
        self.train_epoch_loss = 0
        self.val_epoch_loss = 0
        self.val_epoch_num_correct = 0  # only use when clf
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.network = None
        self.loader = None
        self.tb = None

    def begin_run(self, run, network, train_loader, val_loader=None):

        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.network = network
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tb = SummaryWriter(
            self.log_dir, comment=f"-{run}", filename_suffix=self.filename_suffix
        )

        images, labels = next(iter(self.train_loader))
        self.tb.add_graph(self.network, images.to(getattr(run, "device", "cpu")))

        if self.is_image:
            grid = torchvision.utils.make_grid(images)
            self.tb.add_image("images", grid)

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.train_epoch_loss = 0
        self.val_epoch_loss = 0
        self.val_epoch_num_correct = 0

    def end_epoch(self):

        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        train_loss = self.train_epoch_loss / len(self.train_loader.dataset)
        self.tb.add_scalar("Training Loss", train_loss, self.epoch_count)

        # def send_stats(i, module, input, output):
        #     self.tb.add_scalar(f"layer {i}-mean", output.data.mean())
        #     self.tb.add_scalar(f"layer {i}-stddev", output.data.std())

        # for i, m in enumerate(self.network.children()):
        #     m.register_forward_hook(partial(send_stats, i))

        val_loss = self.val_epoch_loss / len(self.val_loader.dataset)
        self.tb.add_scalar("Validation Loss", val_loss, self.epoch_count)

        if self.is_clf:
            val_accuracy = self.val_epoch_num_correct / len(self.val_loader.dataset)
            self.tb.add_scalar("Accuracy", val_accuracy, self.epoch_count)

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            try:
                self.tb.add_histogram(f"{name}.grad", param.grad, self.epoch_count)
            except:
                self.tb.add_histogram(
                    f"{name}.grad_False", 0, self.epoch_count
                )  # require_grad = False

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results["train_loss"] = train_loss
        results["val_loss"] = val_loss
        if self.is_clf:
            results["accuracy"] = val_accuracy
        results["epoch duration"] = epoch_duration
        results["run duration"] = run_duration
        for k, v in self.run_params._asdict().items():
            results[k] = v
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient="columns")

        clear_output(wait=True)
        display(df)

    def track_loss(self, loss, batch, is_train=True):
        if is_train:
            self.train_epoch_loss += loss.item() * batch[0].shape[0]
        else:
            self.val_epoch_loss += loss.item() * batch[0].shape[0]

    def track_num_correct(self, preds, labels, is_train=False):
        self.val_epoch_num_correct += self._get_num_correct(preds, labels)

    def add_performance(self, performance_metrics):
        for key, value in performance_metrics.items():
            self.tb.add_scalar(key, value, self.epoch_count)

    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def save(self, fileName):

        pd.DataFrame.from_dict(self.run_data, orient="columns").to_csv(f"{fileName}.csv")

        with open(f"{fileName}.json", "w", encoding="utf-8") as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)
