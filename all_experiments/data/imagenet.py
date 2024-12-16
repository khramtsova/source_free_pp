

import os

import torch
from torchvision import transforms
import numpy as np
from torch.utils.data import SubsetRandomSampler, DataLoader, Dataset
from torchvision.datasets import ImageFolder, ImageNet
from data.ImageNetV2_dataset import ImageNetV2Dataset


def ImageNetCorrupted(base_c_path, corruption, corruption_level=0):
    # path to the folder with corrupted images: base_c_path/corruption/corruption_level/
    path = os.path.join(base_c_path, corruption, str(corruption_level))

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    dataset = ImageFolder(path, transform=transform)
    dataset.num_classes = 1000
    return dataset


def ImageNetVal(base_path):
    # path to the folder with corrupted images: base_c_path/corruption/corruption_level/
    path = os.path.join(base_path)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    dataset = ImageFolder(path, transform=transform)
    dataset.num_classes = 1000
    return dataset


def ImageNetSketch(base_path):

    path = os.path.join(base_path)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    dataset = ImageFolder(path, transform=transform)
    dataset.num_classes = 1000
    return dataset


def ImageNetV2(base_path, corruption):
    # 3 variants: matched-frequency, top-images, threshold-0.7
    print("Corruption", corruption)
    if int(corruption) == 0:
        variant = "matched-frequency"
    elif int(corruption) == 1:
        variant = "top-images"
    elif int(corruption) == 2:
        variant = "threshold-0.7"
    else:
        raise NotImplementedError
    path = os.path.join(base_path)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    dataset = ImageNetV2Dataset(location=path,
                                transform=transform,
                                variant=variant)
    dataset.num_classes = 1000
    return dataset

