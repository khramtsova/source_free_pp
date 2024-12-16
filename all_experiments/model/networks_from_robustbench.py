
from robustbench.utils import load_model
import copy
from torch.utils import model_zoo
from torchvision.models.resnet import resnet18, resnet50
from full_project.model.network_utils import Identity



def load_model_robustbench(model_name, model_dir):
    #  ResNet-18 from robustbench
    if model_name == "Modas2021PRIMEResNet18":
        resnet = load_model(model_name=model_name,
                            model_dir=model_dir,
                            threat_model="corruptions",
                            dataset='cifar10')
        resnet.n_outputs = 512
        classifier = copy.deepcopy(resnet.linear)
        del resnet.linear
        resnet.linear = Identity()
    elif model_name == "Standard":
        resnet = load_model(model_name=model_name,
                            model_dir=model_dir,
                            threat_model="corruptions",
                            dataset='cifar10')

        resnet.n_outputs = 640
        classifier = copy.deepcopy(resnet.fc)

        del resnet.fc
        resnet.fc = Identity()

    elif model_name == "resnet18_imagenet":
        # ResNet18
        resnet_weights = model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth',
                                            model_dir=model_dir)
        resnet = resnet18(weights=None)
        resnet.load_state_dict(resnet_weights)
        resnet.n_outputs = 512
        classifier = copy.deepcopy(resnet.fc)
        del resnet.fc
        resnet.fc = Identity()

    elif model_name == "resnet50_imagenet":
        # ResNet50
        resnet_weights = model_zoo.load_url('https://download.pytorch.org/models/resnet50-0676ba61.pth',
                                            model_dir=model_dir)
        resnet = resnet50(weights=None)
        resnet.load_state_dict(resnet_weights)
        resnet.n_outputs = 2048
        classifier = copy.deepcopy(resnet.fc)
        del resnet.fc
        resnet.fc = Identity()
    elif model_name == "Salman2020Do_50_2_Linf":
        # WideResNet 50-2
        resnet = load_model(model_name="Salman2020Do_50_2_Linf",
                            model_dir=model_dir,
                            threat_model="corruptions",
                            dataset='imagenet')
        resnet = resnet.model
        resnet.n_outputs = 2048
        classifier = copy.deepcopy(resnet.fc)
        del resnet.fc
        resnet.fc = Identity()
    else:
        raise NotImplementedError
    return resnet, classifier

