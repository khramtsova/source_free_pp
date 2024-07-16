import time
import torch
import torch.nn.functional as F
from torch.utils import model_zoo
from torch.utils.data import DataLoader
from data.cifar10 import CifarCorrupted, Cifar10_1
from torch import nn

from collections import defaultdict, OrderedDict

from backpack.extensions import BatchGrad, Variance, GGNMP, HMP, PCHMP, DiagGGNMC, BatchDiagGGNExact, DiagGGNExact
from backpack import backpack, extend
from model.custom_optimizer import CGNOptimizer

import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss, Linear
from model.networks_from_robustbench import load_model_robustbench
from model.networks_from_domainbed import load_model_domainbed
from model.networks_custom import load_model_custom
from data.datasets import get_dataset

from utils.logger import Logger
from utils.losses import EntropyLoss
from utils.get_args import get_args
import numpy as np
import argparse
from utils.distances import average_between_dict_list, l2_between_dicts, cos_between_dicts
from utils.distances import get_grad_cos_batch


class CustomMSELoss(torch.nn.Module):
    def __init__(self, num_classes, device):
        super(CustomMSELoss, self).__init__()
        self.num_classes = num_classes
        # self.logsoftmax = torch.nn.LogSoftmax(dim=-1).to(device)
        self.device = device

    def forward(self, logits, labels):
        if labels is None:
            labels = logits.argmax(dim=1)
            pseudo_labels = torch.argmax(logits, dim=1)
            one_hot = torch.nn.functional.one_hot(pseudo_labels, num_classes=self.num_classes)
            # change type from int to float
            labels = one_hot.type(torch.FloatTensor).to(self.device)

        log_softmax = F.log_softmax(logits, dim=1, _stacklevel=5)
        loss = torch.sum(torch.sum(-labels * log_softmax, dim=-1))
        return loss


class Model():
    def __init__(self, model_name="Modas2021PRIMEResNet18",
                 model_dir="/opt/model/", device="cuda:0", num_classes=10):
        super(Model, self).__init__()

        self.model_name = model_name
        self.model_dir = model_dir
        self.device = device
        custom_model_names = ["lenet", "densenet121_camelyon", "fmow_erm", "iwildcam_erm", "rxrx1_erm"]
        if model_name in custom_model_names:
            self.feature_extractor, self.classifier = load_model_custom(model_name, model_dir)
        elif model_name == "resnet50_domainbed":
            self.feature_extractor, self.classifier = load_model_domainbed(model_name, model_dir,
                                                                           input_shape=(3, 224, 224),
                                                                           num_classes=num_classes)
        else:
            self.feature_extractor, self.classifier = load_model_robustbench(model_name, model_dir)

        # change all the relu layers to be not in-place
        for name, module in self.feature_extractor.named_modules():
            if isinstance(module, nn.ReLU) and module.inplace:
                module.inplace = False

        # change all the relu layers to be not in-place
        for name, module in self.classifier.named_modules():
            if isinstance(module, nn.ReLU) and module.inplace:
                module.inplace = False

        # Extend batch normalization layers in the feature extractor
        # for name, module in self.feature_extractor.named_modules():
        #     if isinstance(module, nn.BatchNorm2d):
        #         module = extend(module)

        # use_converter=True replaces all the inplace operations with non-inplace
        self.classifier = extend(self.classifier) # , use_converter=True)
        self.bce_extended = extend(nn.CrossEntropyLoss(reduction="sum"))  #  ))
        self.custom_mse = extend(CustomMSELoss(num_classes=num_classes, device=self.device))
        # define an unsupervised entropy loss
        self.entropy_loss = extend(EntropyLoss())



def get_grad_average(fishr, features, loss_type, batch_size=10000, labels=None):
    if features.shape[0] > batch_size:
        # calculate gradients for each feature split;
        # take the average from the list
        features_split = torch.split(features, batch_size)
        if labels is not None:
            labels_split = torch.split(labels, batch_size)
        grad_mean_list, grad_variance_list, grad_cov_list = [], [], []
        for i, features_split_i in enumerate(features_split):
            if labels is not None:
                labels_subset = labels_split[i]
            fishr.forward_features_through_classifier(features_split_i, loss_type=loss_type, labels=labels_subset)
            grad_mean, grad_variance, grad_cov = fishr.get_grad()
            grad_mean_list.append(grad_mean)
            grad_variance_list.append(grad_variance)
            grad_cov_list.append(grad_cov)
            # print("Done with split")
        grad_mean = average_between_dict_list(grad_mean_list)
        grad_variance = average_between_dict_list(grad_variance_list)
        # grad_cov = average_between_dict_list(grad_cov_list)
        grad_cov = None
    else:
        fishr.forward_features_through_classifier(features, loss_type=loss_type, labels=labels)
        grad_mean, grad_variance, grad_cov = fishr.get_grad()
    return grad_mean, grad_variance, grad_cov
