

import numpy as np
import os
import torch
from backpack import backpack, extend
from backpack.extensions import BatchGrad


def load_features(dataloader, feature_extractor, fname, device):
    """
    Load features from a file if it exists, otherwise create a file and store the features

    :param dataloader: iterable dataset
    :param feature_extractor: the network without the last layer
    :param fname: File name to store the features
    :param device:
    :return: TensorDataset with the features and labels
    """
    if not os.path.exists(fname):
        feature_extractor.eval()
        feature_extractor.to(device)
        print("Creating a file to store the features")
        # create a folder and a file to store the embeddings
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        f = open(fname, 'a')
        f_label = open(fname.replace('.csv', '_label.csv'), 'a')
        for batch_number, data in enumerate(dataloader):
            im, lbl = data[0].to(device), data[1].to(device)
            with torch.no_grad():
                features = feature_extractor(im)
                # save features in a file
                np.savetxt(f, features.detach().cpu().numpy(), delimiter=',')
                np.savetxt(f_label, lbl.detach().cpu().numpy(), delimiter=',')
        f.close(), f_label.close()

    existing_embeddings_np = np.loadtxt(fname, delimiter=',')
    existing_embeddings_torch = torch.Tensor(existing_embeddings_np).type(torch.FloatTensor).to(device)
    existing_labels = np.loadtxt(fname.replace('.csv', '_label.csv'), delimiter=',')

    existing_labels = torch.Tensor(existing_labels).type(torch.LongTensor).to(device)
    # Dataloader to iterate through the embeddings and labels
    dset = torch.utils.data.TensorDataset(existing_embeddings_torch, existing_labels)

    return dset


def get_grad_norm_pytorch(classifier, predicted_prob, target_prob):
    loss = custom_ce_loss(predicted_prob, target_prob)
    loss_summed = torch.sum(loss)
    with backpack(BatchGrad()):
        loss_summed.backward(retain_graph=True)
    batch_grad = classifier.weight.grad_batch
    grad_reshaped = batch_grad.reshape((batch_grad.shape[0], -1))
    grad_norm = torch.norm(grad_reshaped, dim=1)
    return grad_norm  #  newton_grad_norm


def custom_ce_loss(predicted_prob, target_prob):
    loss = -(target_prob*torch.log(predicted_prob+1e-8))  # /N
    return loss


def cov_pytorch(tensor, rowvar=True, bias=False):
    """Estimate a covariance matrix (np.cov)
    """
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()


class GreyToColor(object):
    """Convert Grey Image label to binary
    """

    def __call__(self, image):
        if len(image.size()) == 3 and image.size(0) == 1:
            return image.repeat([3, 1, 1])
        elif len(image.size()) == 2:
            return
        else:
            return image