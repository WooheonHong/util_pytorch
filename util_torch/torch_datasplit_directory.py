from pathlib import Path
import shutil
import os
import random
import re
import numpy as np
from itertools import compress


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

