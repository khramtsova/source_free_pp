

import argparse
import torch
import numpy as np
import random


def get_args():

    # add args  from argparse
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--log_dir', default="./logs/",
                        type=str, help='log directory')
    parser.add_argument('--model_dir', default="/opt/model/",
                        type=str, help='model directory')
    parser.add_argument('--data_dir', default="/opt/data/",
                        type=str, help='data directory')
    parser.add_argument('--dname', default="cifar10-c",
                        type=str, help='cifar10-c or cifar-val or imagenet-c')
    parser.add_argument('--dname_source', default="cifar10-val",
                        type=str, help='cifar10-val')
    parser.add_argument('--loss_type', default="cross_entropy",
                        type=str, help='entropy or cross_entropy or pseudo_labels')
    parser.add_argument('--topk_pseudo', default=100, type=int,
                        help='If pseudo labels are used - how many samples per class to consider')
    parser.add_argument('--model_name', type=str, default="Modas2021PRIMEResNet18")
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=10, type=int)

    # Argument l2_reg; should be False by default, true if we want to use it
    parser.add_argument('--l2_reg', action='store_true')
    parser.add_argument('--fixed_temperature', action='store_true', help="Experiments with fixed temperature")

    # For Mathieu's idea
    parser.add_argument('--corruption', default="", type=str)
    parser.add_argument('--n_augmentations', default=1000, type=int)

    parser.add_argument('--use_gmm', action='store_true', help="Use source data for GMM estimation")
    parser.add_argument('--use_ood_val', action='store_true', help="Use OOD validation set to represent source")
    parser.add_argument('--use_source', action='store_true', help="Use source data for GMM estimation")
    parser.add_argument('--expr_id', default="", type=str, help='Experiment id. Used in the naming of the log file')

    parser.add_argument('--openness', default=1., type=float, help='Percentage of the source validation data to use')

    parser.add_argument('--rand_seed', default=42, type=int, help='Random seed')
    parser.add_argument('--model_dir_v2', default="/opt/model/", type=str,
                        help='Second model directory (needed for AgreeScore baseline)')
    args = parser.parse_args()
    return args


def seed_everything(seed=42):
    # function that set the seed to all the random modules
    # for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return True
