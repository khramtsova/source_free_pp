
import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
from model.network_utils import Identity
from torchvision.models.densenet import densenet121
from torchvision.models.resnet import resnet50


def load_model_custom(model_name, model_dir):
    if model_name == "lenet":
        net = LeNet(num_classes=10)
        model_weights = torch.load(model_dir+"/lenet_mnist/model.pt")
        net.load_state_dict(model_weights)
        net.n_outputs = 1024
        classifier = copy.deepcopy(net.last_layer)
        del net.last_layer
        net.last_layer = Identity()

    elif model_name == "densenet121_camelyon":
        # DenseNet 121 trained on Camelyon17.
        # The weights are taken from the original Wilds paper:
        # https://worksheets.codalab.org/bundles/0x0dc383dbf97a491fab9fb630c4119e3d
        # The reported accuracy on Test OOD is 77.2%
        model_weights = torch.load(model_dir+"/camelyon17_densenet/best_model.pth")
        # currently the weights have a structure model. ...
        # remove model. from the keys
        model_weights["algorithm"] = {k.replace("model.", ""): v for k, v in model_weights["algorithm"].items()}
        net = densenet121(weights=None)
        # Replace the last layer with a Linear layer with 2 classes instead of 1000
        net.classifier = nn.Linear(1024, 2)

        net.load_state_dict(model_weights["algorithm"])
        net.n_outputs = 1024
        classifier = copy.deepcopy(net.classifier)
        del net.classifier
        net.classifier = Identity()
    elif model_name == "fmow_erm":
        model_weights = torch.load(model_dir+"/fmow_erm/fmow_seed_0_epoch_best_model.pth")
        # currently the weights have a structure model. ...
        # remove model. from the keys
        model_weights["algorithm"] = {k.replace("model.", ""): v for k, v in model_weights["algorithm"].items()}
        net = densenet121(weights=None)
        net.classifier = nn.Linear(1024, 62)
        net.load_state_dict(model_weights["algorithm"])
        net.n_outputs = 1024
        classifier = copy.deepcopy(net.classifier)
        del net.classifier
        net.classifier = Identity()

    elif model_name == "iwildcam_erm":
        model_weights = torch.load(model_dir + "/iwildcam_erm/last_model.pth")
        # currently the weights have a structure model. ...
        # remove model. from the keys
        model_weights["algorithm"] = {k.replace("model.", ""): v for k, v in model_weights["algorithm"].items()}
        net = resnet50(weights=None)
        net.fc = nn.Linear(2048, 182)
        net.load_state_dict(model_weights["algorithm"])
        net.n_outputs = 182
        classifier = copy.deepcopy(net.fc)
        del net.fc
        net.fc = Identity()

    elif model_name == "rxrx1_erm":

        model_weights = torch.load(model_dir + "/rxrx1_erm/rxrx1_seed_0_epoch_best_model.pth",)
        # currently the weights have a structure model. ...
        # remove model. from the keys
        model_weights["algorithm"] = {k.replace("model.", ""): v for k, v in model_weights["algorithm"].items()}
        net = resnet50(weights=None)
        net.fc = nn.Linear(2048, 1139)
        net.load_state_dict(model_weights["algorithm"])
        net.n_outputs = 1139
        classifier = copy.deepcopy(net.fc)
        del net.fc
        net.fc = Identity()
    else:
        raise NotImplementedError

    return net, classifier


class LeNet(nn.Module):

    def __init__(self, num_classes=10, **kwargs):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.fc1 = nn.Linear(128 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.extra_head_prior = nn.Linear(1024, 4)
        self.last_layer = nn.Linear(1024, num_classes)

    def forward(self, x, stop="full"):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = conv_out = x.view(x.size(0), -1)
        x = fc1 = F.relu(self.fc1(x))
        x = fc2 = self.fc2(x)
        x = fc2_2 = F.relu(x)
        x = fc3 = self.last_layer(x)

        if stop == "fc1":
            return fc1
        elif stop == "fc2" or stop == "feature":
            return fc2
        elif stop == "fc3":
            return fc3
        elif stop == "full":
            return fc3
        elif stop == "extra_head":
            # x = self.extra_head_prior(conv_out)
            x = self.extra_head_prior(fc2_2)
            return x
        else:
            raise "Unknown stop layer"