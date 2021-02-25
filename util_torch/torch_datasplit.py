"""
Example)
dataset = ESC50(Path.cwd() / "ESC-50/audio")
split = DataSplit(dataset, shuffle=True)
train_loader, val_loader, test_loader = split.get_split(batch_size=bs, num_workers=8)
"""
import shutil
import os
import random
import re
import numpy as np
import logging
from pathlib import Path
from itertools import compress
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler



class DataSplit:
    def __init__(
        self, dataset, test_train_split=0.85, val_train_split=0.18, shuffle=False
    ):
        self.dataset = dataset

        dataset_size = len(dataset)
        self.indices = list(range(dataset_size))
        test_split = int(np.floor(test_train_split * dataset_size))

        if shuffle:
            np.random.shuffle(self.indices)

        train_indices, self.test_indices = (
            self.indices[:test_split],
            self.indices[test_split:],
        )
        train_size = len(train_indices)
        validation_split = int(np.floor((1 - val_train_split) * train_size))

        self.train_indices, self.val_indices = (
            train_indices[:validation_split],
            train_indices[validation_split:],
        )

        self.train_sampler = SubsetRandomSampler(self.train_indices)
        self.val_sampler = SubsetRandomSampler(self.val_indices)
        self.test_sampler = SubsetRandomSampler(self.test_indices)

    def get_train_split_point(self):
        return len(self.train_sampler) + len(self.val_indices)

    def get_validation_split_point(self):
        return len(self.train_sampler)

    @lru_cache(maxsize=4)
    def get_split(self, batch_size=50, num_workers=4):
        logging.debug("Initializing train-validation-test dataloaders")
        self.train_loader = self.get_train_loader(
            batch_size=batch_size, num_workers=num_workers
        )
        self.val_loader = self.get_validation_loader(
            batch_size=batch_size, num_workers=num_workers
        )
        self.test_loader = self.get_test_loader(
            batch_size=batch_size, num_workers=num_workers
        )
        return self.train_loader, self.val_loader, self.test_loader

    @lru_cache(maxsize=4)
    def get_train_loader(self, batch_size=50, num_workers=4):
        logging.debug("Initializing train dataloader")
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=self.train_sampler,
            shuffle=False,
            num_workers=num_workers,
        )
        return self.train_loader

    @lru_cache(maxsize=4)
    def get_validation_loader(self, batch_size=50, num_workers=4):
        logging.debug("Initializing validation dataloader")
        self.val_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=self.val_sampler,
            shuffle=False,
            num_workers=num_workers,
        )
        return self.val_loader

    @lru_cache(maxsize=4)
    def get_test_loader(self, batch_size=50, num_workers=4):
        logging.debug("Initializing test dataloader")
        self.test_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=self.test_sampler,
            shuffle=False,
            num_workers=num_workers,
        )
        return self.test_loader


def make_directory_split(SOURCE=None, SPLIT_SIZE=0.7):
    
    print(SOURCE)
    if SOURCE == None:
        print("파일이 존재하는 경로명 입력")
        print("예시: ESC-50/audio")
        SOURCE = input("")
        SOURCE = Path.cwd() / SOURCE

    path_train = SOURCE / "train"
    path_val = SOURCE / "validation"
    path_test = SOURCE / "test"

    list_no_wav = [f for f in os.listdir(SOURCE) if re.search("^((?!wav).)*$", f)]

    if any([_ in list_no_wav for _ in ["train", "validation", "test"]]):
        print("기존 폴더를 삭제하고 새롭게 생성합니다.")
        for split in [path_train, path_val, path_test]:
            try:
                shutil.rmtree(split)
            except:
                pass
    else:
        print("train, validation, test 폴더를 생성합니다.")

    files = []
    print("Split Data")
    for filename in os.listdir(SOURCE):
        file = str(SOURCE) + "/" + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    try:
        Path.mkdir(path_train)
        Path.mkdir(path_val)
        Path.mkdir(path_test)
    except OSError:
        pass

    training_length = int(len(files) * SPLIT_SIZE)
    validation_length = int(len(files) * 0.15)
    testing_length = int(len(files) - training_length - validation_length)

    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[:training_length]
    validation_set = shuffled_set[
        training_length : (training_length + validation_length)
    ]
    testing_set = shuffled_set[training_length + validation_length :]

    SOURCE = str(SOURCE)
    print(
        "SOURCE: ",
        SOURCE,
        "\n TRAINING",
        SOURCE + "/train",
        "\n VALIDATION",
        SOURCE + "/validation",
        "\n test",
        SOURCE + "/test",
        "\n ",
        len(files),
    )
    print("training_length:", training_length)
    print("validation_length:", validation_length)
    print("testing_length:", testing_length)

    for filename in training_set:
        this_file = SOURCE + "/" + filename
        destination = SOURCE + "/train" + "/" + filename
        shutil.copy(this_file, destination)

    for filename in validation_set:
        this_file = SOURCE + "/" + filename
        destination = SOURCE + "/validation" + "/" + filename
        shutil.copy(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + "/" + filename
        destination = SOURCE + "/test" + "/" + filename
        shutil.copy(this_file, destination)

