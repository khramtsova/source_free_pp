
import torch.multiprocessing
from torchvision import transforms, datasets
import torchvision.transforms.functional as F
from data.imagenet import ImageNetCorrupted, ImageNetVal, ImageNetSketch, ImageNetV2
from data.cifar10 import CifarCorrupted, Cifar10_1
from data.cifar10 import CifarCorrupted_Augmented
from data.synth import SynthDigit
from data.domain_bed_dsets import DomainNet, PACS, OfficeHome, VLCS, TerraIncognita
from data.domain_bed_utils.misc_utils import split_dataset, seed_hash

from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset
from wilds.datasets.iwildcam_dataset import IWildCamDataset
from wilds.datasets.rxrx1_dataset import RxRx1Dataset


def get_dataset(d_name, data_dir, corruption, custom_transform=None, use_ood_val=False):
    dset = {}
    if d_name == "cifar10":
        source_transform = transforms.Compose([ transforms.ToTensor(), transforms.Resize((32, 32), antialias=True)])
        source_val = datasets.CIFAR10("{}/CIFAR10/".format(data_dir),
                                      train=False, download=False, transform=source_transform)
        num_classes = 10
        if corruption == "cifar10-c":
            for corruption in ["gaussian_noise", "shot_noise", "impulse_noise",
                                "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
                                "snow", "frost", "fog",
                                "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"
                                ]:
                for corr in range(5):
                    corr_id = "{}_{}".format(corruption, corr)
                    dset[corr_id] = CifarCorrupted(base_c_path="{}/CIFAR-10-C/".format(data_dir),
                                                             corruption=corruption,
                                                             corruption_level=corr,
                                                             custom_transform=custom_transform)

        elif corruption == "cifar10-1":
            dset["cifar10-1"] = Cifar10_1("{}/CIFAR-10.1/".format(data_dir))


    elif d_name == "digits":
        test_transform = transforms.Compose([transforms.Resize((32, 32)),
            transforms.ToTensor(), GreyToColor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        source_val = datasets.MNIST(root=data_dir, train=False, download=False, transform=test_transform)
        num_classes = 10
        dset["svhn"] = datasets.SVHN(root=data_dir + "/SVHN/", split='test',
                                     download=False, transform=test_transform)
        dset["usps"] = datasets.USPS(root=data_dir + "/USPS/", train=False,
                                        download=False, transform=test_transform)
        dset["synth"] = SynthDigit(root=data_dir + "/SynthDigits/", train=False,
                                        transform=test_transform)

    elif d_name == "imagenet":
        torch.multiprocessing.set_sharing_strategy('file_system')

        source_val = ImageNetVal("{}/ImageNet-Val/".format(data_dir))
        num_classes = 1000

        if corruption == "imagenet-sketch":
            dset["imagenet-sketch"] = ImageNetSketch("{}/ImageNet-Sketch/sketch/".format(data_dir))
        elif corruption == "imagenet-v2":
            for corr in range(3):
                corr_id = "imagenet-v2_{}".format(corr)
                dset[corr_id] = ImageNetV2(base_path="{}/ImageNet-V2/".format(data_dir), corruption=corr)
        else:
            raise NotImplementedError
        # dset["imagenet-c"] = ImageNetCorrupted("{}/ImageNet-C/".format(data_dir), corruption, corr_level + 1)

    elif any(ds in d_name for ds in ["pacs", "domain_net", "office_home", "vlcs", "terra_incognita"]):

        if "pacs" in d_name:
            dset_domain_bed = PACS(data_dir, test_envs=corruption)
            corr_id = "pacs-test_{}".format(corruption)
        elif "domain_net" in d_name:
            dset_domain_bed = DomainNet(data_dir, test_envs=corruption)
            corr_id = f"domain_net-test-{corruption}"
        elif "office_home" in d_name:
            dset_domain_bed = OfficeHome(data_dir, test_envs=corruption)
            corr_id = f"office_home-test-{corruption}"
        elif "vlcs" in d_name:
            dset_domain_bed = VLCS(data_dir, test_envs=corruption)
            corr_id = f"vlcs-test-{corruption}"
        elif "terra_incognita" in d_name:
            dset_domain_bed = TerraIncognita(data_dir, test_envs=corruption)
            corr_id = f"terra_incognita-test-{corruption}"

        num_classes = dset_domain_bed.num_classes

        dset[corr_id] = dset_domain_bed.datasets[int(corruption)]

        out_splits = []
        for env_i, env in enumerate(dset_domain_bed):
            # Exclude the test environment
            if env_i != int(corruption):
                out, in_ = split_dataset(env,
                                         int(len(env) * 0.2),  # args.holdout_fraction = 0.2
                                         seed_hash(0, env_i))  # trial_seed = 0
                # print("Env: {}, Sample size: {}".format(env_i, len(out)))
                out_splits.append(out)
        # combine the validation splits into one dataset
        source_val = torch.utils.data.ConcatDataset(out_splits)

    elif "camelyon" in d_name:
        # Get the test set
        dataset = Camelyon17Dataset(root_dir="{}/Wilds/".format(data_dir), download=False)
        transform = transforms.Compose([transforms.Resize((96, 96)), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        dset["camelyon-test"] = dataset.get_subset("test", transform=transform)

        if use_ood_val:
            source_val = dataset.get_subset("val", transform=transform)
        else:
            source_val = dataset.get_subset("id_val", transform=transform)
        num_classes = 2

    elif "fmow" in d_name:
        dataset = FMoWDataset(root_dir="{}/Wilds/".format(data_dir), download=False)
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if use_ood_val:
            source_val = dataset.get_subset("val", transform=transform)
        else:
            source_val = dataset.get_subset("id_val", transform=transform)
        num_classes = 62
        dset["fmow-test"] = dataset.get_subset("test", transform=transform)


    elif "iwildcam" in d_name:
        dataset = IWildCamDataset(root_dir="{}/Wilds/".format(data_dir), download=False)
        transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])])

        if use_ood_val:
            source_val = dataset.get_subset("val", transform=transform)
        else:
            source_val = dataset.get_subset("id_val", transform=transform)
        num_classes = 182
        dset["iwildcam-test"] = dataset.get_subset("test", transform=transform)


    elif "rxrx1" in d_name:
        def standardize(x: torch.Tensor) -> torch.Tensor:
            mean = x.mean(dim=(1, 2))
            std = x.std(dim=(1, 2))
            std[std == 0.] = 1.
            return F.normalize(x, mean, std)
        t_standardize = transforms.Lambda(lambda x: standardize(x))
        transform = transforms.Compose([transforms.ToTensor(), t_standardize])
        dataset = RxRx1Dataset(root_dir="{}/Wilds/".format(data_dir),
                                        download=False)
        dset["rxrx1-test"] = dataset.get_subset("test", transform=transform)
        num_classes = 1139
        raise "RxRx1 source dset not implemented yet"
    else:
        raise NotImplementedError
    return dset, source_val, num_classes


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