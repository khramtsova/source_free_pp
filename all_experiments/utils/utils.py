
import torch
import torch.nn as nn
from backpack import backpack, extend
from backpack.extensions import BatchGrad
import numpy as np

from utils.ATC_helpers.utils import softmax_numpy, inverse_softmax_numpy, get_entropy, find_threshold
from utils.losses import custom_ce_loss, custom_ce_no_log, cauchy_activation


def cov_pytorch(tensor, rowvar=True, bias=False):
    """Estimate a covariance matrix (np.cov)
    """
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()


def set_bn_to_train(net):
    # function that iterates through the layers of the network
    # and change all the batch norm layers to train mode
    for layer in net.modules():
        if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
            layer.train()


def get_grad(prob, labels, features):
    grad = (prob - labels).unsqueeze(1) * features.unsqueeze(2)
    # Reshape to have the batch size as the first dimension, and the rest as the second
    grad_reshaped = grad.reshape((grad.shape[0], -1))
    grad_norm = torch.norm(grad_reshaped, dim=1)
    return grad, grad_norm


def get_grad_bias(prob, labels, features=None):
    grad = prob - labels
    # Reshape to have the batch size as the first dimension, and the rest as the second
    grad_norm = torch.norm(grad, dim=1)
    return grad, grad_norm


def get_grad_norm_pytorch(fishr, predicted_prob, target_prob):
    # loss = custom_ce_no_log(predicted_prob, target_prob)
    loss = custom_ce_loss(predicted_prob, target_prob)
    # per sample loss
    loss_per_sample = torch.sum(loss, dim=1)
    loss_summed = torch.sum(loss)
    with backpack(BatchGrad()):
        loss_summed.backward(retain_graph=True)
    # print("After backward")

    # nn.utils.clip_grad_norm_(fishr.classifier.parameters(), max_norm=1000)

    batch_grad = fishr.classifier.weight.grad_batch
    grad_reshaped = batch_grad.reshape((batch_grad.shape[0], -1))
    grad_norm = torch.norm(grad_reshaped, dim=1)
    #fishr.classifier.bias = None
    bias_grad = fishr.classifier.bias.grad_batch
    grad_bias_reshaped = bias_grad.reshape((bias_grad.shape[0], -1))
    grad_norm_bias = torch.norm(grad_bias_reshaped, dim=1)

    return loss_per_sample, batch_grad, grad_norm, grad_norm_bias  #  newton_grad_norm


def get_entropy_and_probs(probs, labels, calibration=True):
    pred_idx = np.argmax(probs, axis=-1)
    source_acc = np.mean(pred_idx == labels) * 100.
    if not calibration:
        pred_probab = np.max(probs, axis=-1)
        pred_indx = np.argmax(probs, axis=-1)
        entropy = get_entropy(probs)
        return entropy, pred_indx, pred_probab
    probs_calibrated = softmax_numpy(inverse_softmax_numpy(probs))
    pred_indx_calibrated = np.argmax(probs_calibrated, axis=-1)
    pred_probab_calibrated = np.max(probs_calibrated, axis=-1)
    calib_entropy = get_entropy(probs_calibrated)
    return calib_entropy, pred_indx_calibrated, pred_probab_calibrated
