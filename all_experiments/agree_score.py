import os
import torch
from torch.utils.data import DataLoader
from full_project.data.datasets import get_dataset
import numpy as np
import pandas as pd
from all_experiments.utils.get_args import get_args, seed_everything
from utils.logger import Logger
from main import Model
from wilds.common.data_loaders import get_eval_loader



def forward_and_eval(dataloader, fishr, fishr_v2, logger):
    fishr.feature_extractor.eval(), fishr.classifier.eval()
    fishr.feature_extractor.to(device), fishr.classifier.to(device)

    fishr_v2.feature_extractor.eval(), fishr_v2.classifier.eval()
    fishr_v2.feature_extractor.to(device), fishr_v2.classifier.to(device)
    for batch_number, data in enumerate(dataloader):
        im, lbl = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            features_m1, features_m2 = fishr.feature_extractor(im), fishr_v2.feature_extractor(im)
            logits_m1, logits_m2 = fishr.classifier(features_m1), fishr_v2.classifier(features_m2)
            prob_m1, prob_m2 = torch.softmax(logits_m1, dim=1), torch.softmax(logits_m2, dim=1)
            pred_class_m1, pred_class_m2 = torch.argmax(logits_m1, dim=1), torch.argmax(logits_m2, dim=1)

            # Log the results
            for indx, label in enumerate(lbl):
                logger.log_custom("labels", label.detach().cpu().item())
                logger.log_custom("pred_m1", pred_class_m1[indx].detach().cpu().item())
                logger.log_custom("pred_m2", pred_class_m2[indx].detach().cpu().item())
                logger.add_record_to_log_file()
    return True


def calculate_stats(args):

    if "domain_net" in args.dname:
        args.model_name = "resnet50_domainbed"
        # args.corruption is the index of corruption in this case
        # Example:  "./model/model_weights/resnet_domainbed/3/model_test_envs[3].pkl"
        args.model_dir = "{}/resnet50_domain_net/{}/model_test_envs[{}].pkl".format(args.model_dir,
                                                                                    args.corruption,
                                                                                    args.corruption)
    elif "pacs" in args.dname:
        args.model_name = "resnet50_domainbed"
        args.model_dir_v2 = "{}/agree_score/resnet50_pacs/ERM/{}/model.pkl".format(args.model_dir, args.corruption)
        args.model_dir = "{}/resnet50_pacs/{}/model.pkl".format(args.model_dir, args.corruption)
    elif "vlcs" in args.dname:
        args.model_name = "resnet50_domainbed"
        args.model_dir_v2 = "{}/agree_score/resnet50_vlcs/ERM/{}/model.pkl".format(args.model_dir, args.corruption)
        args.model_dir = "{}/resnet50_vlcs/ERM/{}/model.pkl".format(args.model_dir, args.corruption)
    elif "office_home" in args.dname:
        args.model_name = "resnet50_domainbed"
        args.model_dir_v2 = "{}/agree_score/resnet50_office_home/ERM/{}/model.pkl".format(args.model_dir, args.corruption)
        args.model_dir = "{}/resnet50_office_home/ERM/{}/model.pkl".format(args.model_dir, args.corruption)
    elif "terra_incognita" in args.dname:
        args.model_name = "resnet50_domainbed"
        args.model_dir_v2 = "{}/agree_score/resnet50_terra_incognita/ERM/{}/model.pkl".format(args.model_dir, args.corruption)
        args.model_dir = "{}/resnet50_terra_incognita/ERM/{}/model.pkl".format(args.model_dir, args.corruption)

    target_dset, source_dset, num_classes = get_dataset(args.dname, args.data_dir, args.corruption,
                                                        use_ood_val=args.use_ood_val)

    for t_dset_name in target_dset.keys():
        t_set = target_dset[t_dset_name]
        if "camelyon" in args.dname:
            dataloader_target = get_eval_loader("standard", t_set, batch_size=args.batch_size)
        else:
            dataloader_target = DataLoader(t_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

        fishr = Model(model_name=args.model_name, model_dir=args.model_dir, device=device, num_classes=num_classes)
        fishr_2 = Model(model_name=args.model_name, model_dir=args.model_dir_v2, device=device, num_classes=num_classes)

        log_dir = os.path.join(args.log_dir, args.dname)
        logger = Logger(log_dir + "/agree_score/", f"agree_score_{t_dset_name}_{args.corruption}.csv")
        print(f"{log_dir}/features/features_{t_dset_name}_{args.corruption}.csv")
        forward_and_eval(dataloader_target, fishr, fishr_2, logger=logger)
        print(f"Done with {t_dset_name} and {args.corruption}")

    return True


def eval_agree_score(args):
    # read csv file

    path_to_folder = f"{args.log_dir}/{args.dname}/agree_score/"
    gt_acc, agree_score_acc, mae = [], [], []
    sorted_files = sorted(os.listdir(path_to_folder))
    for file in sorted_files:
        print(file)
        df = pd.read_csv(path_to_folder + file)
        agree_score = sum(df["pred_m1"] == df["pred_m2"]) / len(df["labels"]) * 100
        gt = sum(df["labels"] == df["pred_m1"]) / len(df["labels"]) * 100

        # print(sum(df["pred_m2"] == df["labels"]) / len(df["labels"]) * 100)
        print(abs(agree_score - gt))
        gt_acc.append(gt)
        agree_score_acc.append(agree_score)
        mae.append(np.mean(abs(agree_score - gt)))

    print(args.dname, round(np.mean(mae), 2) ,   np.mean(gt_acc), mae)


if __name__ == '__main__':
    seed_everything(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device", device)
    args = get_args()
    # Store the results

    # calculate_stats(args)
    eval_agree_score(args)