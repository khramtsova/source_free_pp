
import os

import torch
from torchvision import transforms
import numpy as np
from torch.utils.data import SubsetRandomSampler, DataLoader, Dataset


class CifarCorrupted(Dataset):
    def __init__(self, base_c_path, corruption, custom_transform=None, corruption_level=0):
        self.custom_transform = custom_transform
        self.transform_post_custom = transforms.Compose([
                # transforms.ToTensor(),
                transforms.Resize((32, 32), antialias=True),
            ])

        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((32, 32), antialias=True),
            ])

        self.images = np.load(os.path.join(base_c_path, corruption + '.npy'))
        self.labels = torch.LongTensor(np.load(os.path.join(base_c_path, 'labels.npy')))
        self.num_classes = 10
        self.classes = [i for i in range(self.num_classes)]

        self.images = self.images[corruption_level * 10000:(corruption_level + 1) * 10000]
        self.labels = self.labels[corruption_level * 10000:(corruption_level + 1) * 10000]
        """
        if level5_corruption_only:
            # Should be -10000
            self.images = self.images[-10000:]
            self.labels = self.labels[-10000:]
        """

    def __getitem__(self, index):
        # imgs = torch.from_numpy(self.images[index]).float()
        imgs = self.images[index]
        labels = self.labels[index]
        if self.custom_transform:
            imgs = self.custom_transform(imgs)
            imgs = self.transform_post_custom(imgs)
        else:
            imgs = self.transform(imgs)
        return imgs, labels

    def __len__(self):
        return len(self.labels)


class CifarCorrupted_Augmented(Dataset):
    def __init__(self, base_c_path, corruption, custom_transform=None, corruption_level=0, num_augmentations=20):
        self.num_augmentations = num_augmentations
        self.custom_transform = custom_transform
        self.transform_post_custom = transforms.Compose([
                # transforms.ToTensor(),
                transforms.Resize((32, 32), antialias=True),
            ])

        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((32, 32), antialias=True),
            ])

        self.images = np.load(os.path.join(base_c_path, corruption + '.npy'))
        self.labels = torch.LongTensor(np.load(os.path.join(base_c_path, 'labels.npy')))
        self.num_classes = 10
        self.classes = [i for i in range(self.num_classes)]

        self.images = self.images[corruption_level * 10000:(corruption_level + 1) * 10000]
        self.labels = self.labels[corruption_level * 10000:(corruption_level + 1) * 10000]

    def __getitem__(self, index):
        # imgs = torch.from_numpy(self.images[index]).float()
        img = self.images[index]
        labels = self.labels[index]
        # return original image and 20 augmented images
        imgs = [self.transform(img)]
        for i in range(self.num_augmentations):
            self.custom_transform.set_transform_indx(i)
            imgs.append(self.transform_post_custom(self.custom_transform(img)))
        imgs = torch.stack(imgs)
        return imgs, labels

    def __len__(self):
        return len(self.labels)



class Cifar10_1(Dataset):
    def __init__(self, data_dir):
        self.images = np.load(os.path.join(data_dir, 'cifar10.1_v6_data.npy'))
        # self.images = np.transpose(images, (0, 3, 1, 2))
        self.labels = torch.LongTensor(np.load(os.path.join(data_dir, 'cifar10.1_v6_labels.npy')))
        self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32), antialias=True),
        ])
        self.num_classes = 10

    def __getitem__(self, index):
        # imgs = torch.from_numpy(self.images[index]).float()
        imgs = self.images[index]
        labels = self.labels[index]
        if self.transform:
            imgs = self.transform(imgs)
        return imgs, labels

    def __len__(self):
        return len(self.labels)