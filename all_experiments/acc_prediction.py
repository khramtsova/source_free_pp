import torch
from torch.utils.data import DataLoader
from data.aug_lib import CustomTransform

from full_project.data.datasets import get_dataset

from all_experiments.utils.get_args import get_args, seed_everything
from all_experiments.utils.distances import l2_between_dicts
from utils.logger import Logger
from main import Model, get_grad_average
import time

from backpack.extensions import BatchGrad
from backpack import backpack
import gc
from collections import OrderedDict
from all_experiments.utils.distances import norm_of_the_dict, sign_alignment_between_dicts, rms_of_the_dict


def prediction_idea_1(args):

    # First idea:
    # For each sample, take the gradient with respect to the most probable prediciton
    #  vs the gradient with respect to all the other predictions
    #  See which gradient is more allighed with the mean gradient

    logger = Logger(args.log_dir, "logs_{}_{}_{}_{}.csv".format(args.dname, args.model_name,
                                                                args.loss_type, args.corruption))

    fishr = Model(model_name=args.model_name, model_dir=args.model_dir, device=device)

    # Load the target dataset
    target = get_dataset(args.dname, args.data_dir, args.corruption, corr_level=4)
    # target = torch.utils.data.Subset(target, list(range(200)))
    # target.num_classes = 10
    # print("Target Dataset size", len(target))
    dataloader_target = DataLoader(target, batch_size=args.batch_size,
                                   shuffle=True, num_workers=10)
    acc, features, labels, predictions, logits, sampled_labels = fishr.get_acc_and_features(dataloader_target)
    print("Accuracy", acc)

    # Get means for the whole unaugmented dataset
    grad_mean_whole_dset, _, _ = get_grad_average(fishr=fishr,
                                                  features=features,
                                                  loss_type=args.loss_type)
    grad_mean_per_class = {}
    for cls in range(10):  # range(target.classes):
        # get index where prediction is cls
        features_per_class = features[predictions == cls]
        grad_mean_subset, _, _ = get_grad_average(fishr=fishr,
                                                  features=features_per_class,
                                                  loss_type=args.loss_type)
        grad_mean_per_class[cls] = grad_mean_subset

    print(grad_mean_per_class[0])
    print(grad_mean_per_class[1])
    """
    logger = Logger(args.log_dir, "stats_{}_{}_{}_{}.csv".format(args.dname, args.model_name,
                                                            args.loss_type, args.corruption))
    # for each class compute the dot product with all the other classes
    for cls in range(10):
        for cls_2 in range(10):
            dot_prod = dot_between_dicts(grad_mean_per_class[cls], grad_mean_per_class[cls_2])
            logger.log_custom("dot_prod_per_class_{}_{}".format(cls, cls_2), dot_prod.cpu().detach().numpy())

    # for each class compute the dot product with the whole dataset
    for cls in range(10):
        dot_prod = dot_between_dicts(grad_mean_per_class[cls], grad_mean_whole_dset)
        logger.log_custom("dot_prod_whole_dset_{}".format(cls), dot_prod.cpu().detach().numpy())
    logger.add_record_to_log_file()
    return

    """
    # For each target sample - calculate the gradients with respect to different predictions
    # Check if the most probable prediction is better alligned with the mean gradient
    #  than the other predictions

    fishr.classifier.train()
    features = features.to(device)

    temp = 0
    for sample, gt_label in zip(features, labels):

        # resize the sample to have the first dimention of 1
        sample = sample.unsqueeze(0)
        # print(sample.shape)
        # for each sample, change the label to be one of the other labels
        all_labels = torch.arange(10).to(device)
        # print(labels)

        logits = fishr.classifier(sample)

        predicted_label = logits.argmax(dim=1)[0].cpu().detach().numpy()

        for label in all_labels:

            # zero the gradients
            fishr.classifier.zero_grad()
            # print(logits, label.unsqueeze(0))
            # calculate the loss with respect to this label
            loss = fishr.bce_extended(logits, label.unsqueeze(0))
            loss.backward(retain_graph=True)

            full_grad = OrderedDict()
            for name, param in fishr.classifier.named_parameters():
                full_grad[name] = param.grad
            # Calculate gradient dot product with the mean gradient
            dot_prod = sign_alignment_between_dicts(full_grad, grad_mean_per_class[label.item()])
            #dot_prod_2 = dot_between_dicts(full_grad, grad_mean_whole_dset)
            logger.log_custom("sample_indx", temp)
            logger.log_custom("sign_per_class", dot_prod)
            #logger.log_custom("dot_prod_whole_dset", dot_prod_2.cpu().detach().numpy())
            logger.log_custom("class", label.item())
            logger.log_custom("gt_label", gt_label.cpu().detach().numpy())
            logger.log_custom("predicted_label", predicted_label)
            logger.add_record_to_log_file()
            # Calculate l2 distance between the full gradient for this sample and the mean gradient
            # grad_dist_to_full = l2_between_dicts(full_grad, grad_mean_whole_dset)
            # grad_dist_to_class = l2_between_dicts(full_grad, grad_mean_per_class[label.item()])
            # print ("Gradient norm", norm_of_the_dict(full_grad).item())
            # print("Label", label.item(), "Loss", loss.item())
            #print("Grad dist to full", grad_dist_to_full)
            #print("Grad dist to class", grad_dist_to_class)

        # print("Predicted label", predicted_label)
        # print("GT label", gt_label.item())
        temp += 1
        # print temp every 100 samples
        if temp % 100 == 0:
            print(temp)
            # raise

    print("Done")

    return


def get_augmented_dloader(args):
    # Load the augmented dataset
    transform_list = ['cutout', 'auto_contrast',
                      'contrast', 'brightness',
                      'equalize', 'sharpness',
                      'solarize', 'color', 'posterize',
                      'shear_x', 'shear_y',
                      'translate_x', 'translate_y']
    custom_transform = CustomTransform(n_transforms=2, level=-1,
                                       transform_list=transform_list,
                                       n_datasets=args.n_augmentations)
    # Change the data transformation
    custom_transform.set_transform_indx(0)
    target_augmented = get_dataset("cifar10-c-augmet",
                                   args.data_dir,
                                   args.corruption, corr_level=4,
                                   custom_transform=custom_transform,
                                   num_augmentations=args.n_augmentations)
    dataloader_target_augmented = DataLoader(target_augmented,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=10)
    return dataloader_target_augmented


def prediction_idea_2_full_net(args):
    logger = Logger(args.log_dir, "logs_{}_{}_{}_{}.csv".format(args.dname, args.model_name,
                                                                args.loss_type, args.corruption))

    fishr = Model(model_name=args.model_name, model_dir=args.model_dir, device=device)

    # Load the target dataset
    target = get_dataset(args.dname, args.data_dir, args.corruption, corr_level=4)
    #target = torch.utils.data.Subset(target, list(range(200)))
    #target.num_classes = 10

    # Get the gradient with respect to the most probable prediction
    # fishr.classifier = fishr.classifier.to(device)
    fishr.classifier.eval()
    fishr.feature_extractor.eval()
    temp = 0
    # for sample, gt_label in zip(features, labels):
    for image, gt_label in target:

        # zero the gradients
        fishr.feature_extractor.zero_grad()
        fishr.classifier.zero_grad()

        # print(sample.shape)
        # clone the sample 10 times
        image = image.unsqueeze(0)
        sample = image.repeat(target.num_classes, 1, 1, 1)
        # print(sample.shape)
        # for each sample, change the label to be one of the other labels
        all_labels = torch.arange(target.num_classes).to(device)
        # print(labels)
        _, predicted_label = fishr.forward_features_through_classifier(sample,
                                                                       loss_type=args.loss_type,
                                                                       labels=all_labels)
        predicted_label = predicted_label[0].cpu().detach().numpy()
        logger.log_custom("gt_label", gt_label.cpu().detach().numpy())
        logger.log_custom("predicted_label", predicted_label)
        # grad_mean, grad_variance, grad_cov = fishr.get_grad()
        for name, param in fishr.feature_extractor.named_parameters():
            # if it is a linear layer or batch norm layer
            if "bn" in name:
                variance, mean = torch.var_mean(param.grad_batch, dim=0)
                logger.log_custom("grad_mean_{}".format(name), torch.norm(mean).cpu().detach().numpy())
                logger.log_custom("grad_var_{}".format(name), torch.norm(variance).cpu().detach().numpy())
        for name, param in fishr.classifier.named_parameters():
            variance, mean = torch.var_mean(param.grad_batch, dim=0)
            logger.log_custom("grad_mean_{}".format(name), torch.norm(mean).cpu().detach().numpy())
            logger.log_custom("grad_var_{}".format(name), torch.norm(variance).cpu().detach().numpy())

        logger.add_record_to_log_file()
        temp += 1
        # print temp every 100 samples
        if temp % 100 == 0:
            print(temp)
    print("Done")
    return


def prediction_idea_2(args):
    logger = Logger(args.log_dir, "logs_{}_{}_{}_{}.csv".format(args.dname, args.model_name,
                                                                args.loss_type, args.corruption))

    fishr = Model(model_name=args.model_name, model_dir=args.model_dir, device=device)

    # Load the target dataset
    target = get_dataset(args.dname, args.data_dir, args.corruption, corr_level=4)
    #target = torch.utils.data.Subset(target, list(range(200)))
    #target.num_classes = 10

    # print("Target Dataset size", len(target))
    dataloader_target = DataLoader(target, batch_size=args.batch_size,
                                   shuffle=True, num_workers=10)
    acc, features, labels, predictions, logits, sampled_labels = fishr.get_acc_and_features(dataloader_target)
    print("Accuracy", acc)

    # Get the gradient with respect to the most probable prediction
    # fishr.classifier = fishr.classifier.to(device)
    fishr.classifier.train()
    features = features.to(device)

    temp = 0
    batch_size = 64
    for sample, gt_label in zip(features, labels):
        # zero the gradients
        fishr.classifier.zero_grad()
        logits = fishr.classifier(sample)
        predicted_label = logits.argmax().cpu().detach().numpy()

        # print(sample.shape)
        # clone the sample 10 times
        sample = sample.repeat(target.num_classes, 1)
        # print(sample.shape)
        # for each sample, change the label to be one of the other labels
        all_labels = torch.arange(target.num_classes).to(device)
        # print(labels)

        grad_mean, grad_var, _ = get_grad_average(fishr=fishr,
                                                  features=sample,
                                                  loss_type=args.loss_type,
                                                  batch_size=128,
                                                  labels=all_labels)

        grad_mean_over_var = OrderedDict()
        # for all the parameters of the layer (aka weight and bias)
        for key in grad_mean.keys():
            grad_mean_over_var[key] = grad_mean[key] / grad_var[key]
        print(torch.norm(grad_mean[key]).cpu().detach().numpy())
        raise
        for key in grad_mean.keys():
            logger.log_custom("grad_mean_{}".format(key), torch.norm(grad_mean[key]).cpu().detach().numpy())
            logger.log_custom("grad_var_{}".format(key), torch.norm(grad_var[key]).cpu().detach().numpy())
            logger.log_custom("grad_mean_over_var_{}".format(key),
                              torch.norm(grad_mean_over_var[key]).cpu().detach().numpy())

        logger.log_custom("gt_label", gt_label.cpu().detach().numpy())
        logger.log_custom("predicted_label", predicted_label)
        logger.log_custom("if_prediction_is_correct", gt_label == predicted_label)
        logger.add_record_to_log_file()

        temp += 1
        # print temp every 100 samples
        if temp % 100 == 0:
            print(temp)
    print("Done")
    return


def prediction_idea_2_1(args):
    logger = Logger(args.log_dir, "logs_{}_{}_{}_{}.csv".format(args.dname, args.model_name,
                                                                args.loss_type, args.corruption))

    fishr = Model(model_name=args.model_name, model_dir=args.model_dir, device=device)

    dataloader_target_augmented = get_augmented_dloader(args)
    fishr.classifier.train()
    temp = 0

    fishr.feature_extractor.to(device)
    fishr.classifier.to(device)
    num_classes = 10
    t1 = time.time()
    for sample, gt_label in dataloader_target_augmented:

        gt_label = gt_label[0]

        # the first dimention is 1 - remove it
        sample = sample.squeeze(0).to(device)
        # now the first dimention is the number of augmentations + 1

        # calculate the gradient
        features = fishr.feature_extractor(sample)
        features = features.repeat(num_classes, 1)

        # prediction
        logits = fishr.classifier(features)

        # Generate 10 labels for each augmented sample
        all_labels = torch.arange(num_classes).to(device)
        all_labels = all_labels.repeat(args.n_augmentations+1, 1).transpose(0, 1).flatten()

        # calculate the loss with respect to this label
        loss = fishr.bce_extended(logits, all_labels).sum()

        # calculate the gradient
        gc.collect()
        with backpack(BatchGrad()):  # , Variance()
            loss.backward()
        grad_mean, grad_var, _ = fishr.get_grad()

        grad_mean_over_var = OrderedDict()
        # for all the parameters of the layer (aka weight and bias)
        for key in grad_mean.keys():
            grad_mean_over_var[key] = grad_mean[key] / grad_var[key]

        predicted_label = logits.argmax(dim=1)[0].cpu().detach().numpy()

        logger.log_custom("grad_mean_rms", rms_of_the_dict(grad_mean).cpu().detach().numpy())
        logger.log_custom("grad_var_rms", rms_of_the_dict(grad_var).cpu().detach().numpy())
        logger.log_custom("grad_mean_over_var_rms", rms_of_the_dict(grad_mean_over_var).cpu().detach().numpy())

        logger.log_custom("grad_mean_norm", norm_of_the_dict(grad_mean).cpu().detach().numpy())
        logger.log_custom("grad_var_norm", norm_of_the_dict(grad_var).cpu().detach().numpy())
        logger.log_custom("grad_mean_over_var_norm", norm_of_the_dict(grad_mean_over_var).cpu().detach().numpy())

        logger.log_custom("gt_label", gt_label.cpu().detach().numpy())
        logger.log_custom("predicted_label", predicted_label)
        logger.log_custom("if_prediction_is_correct", gt_label == predicted_label)
        logger.add_record_to_log_file()

        temp += 1
        # print temp every 100 samples
        if temp % 100 == 0:
            print("Step:{}, Time Elapsed:{}".format(temp, time.time() - t1))

    print("Done")
    return



def prediction_idea_3(args):
    logger = Logger(args.log_dir, "logs_{}_{}_{}_{}.csv".format(args.dname, args.model_name,
                                                                args.loss_type, args.corruption))
    fishr = Model(model_name=args.model_name, model_dir=args.model_dir, device=device)

    target = get_dataset("cifar10-c", args.data_dir,  args.corruption, corr_level=4)
    # target = torch.utils.data.Subset(target, list(range(200)))
    # target.num_classes = 10
    # print("Target Dataset size", len(target))
    dataloader_target = DataLoader(target, batch_size=args.batch_size,
                                   shuffle=True, num_workers=10)
    acc, features, labels, predictions, logits, sampled_labels = fishr.get_acc_and_features(dataloader_target)

    # Get variance for the whole unaugmented dataset
    _, grad_variance_mean, _ = get_grad_average(fishr=fishr,
                                          features=features,
                                          loss_type=args.loss_type)

    grad_variance_per_class = {}
    for cls in range(10):  # range(target.classes):
        # get index where prediction is cls
        # global_indx_per_class = np.where(predictions == cls)[0]
        # logits_per_class = logits[predictions == cls]
        features_per_class = features[predictions == cls]
        _, grad_variance_subset, _ = get_grad_average(fishr=fishr,
                                                                     features=features_per_class,
                                                                     loss_type=args.loss_type)
        grad_variance_per_class[cls] = grad_variance_subset
    # print(grad_variance_per_class)

    del features, labels, predictions, logits, sampled_labels
    gc.collect()

    dataloader_target_augmented = get_augmented_dloader(args)
    temp = 0
    fishr.feature_extractor.eval()
    t1 = time.time()
    for sample, gt_label in dataloader_target_augmented:
        # the first dimention is 1 - remove it
        sample = sample.squeeze(0).to(device)
        # now the first dimention is the number of augmentations + 1

        # calculate the gradient
        features = fishr.feature_extractor(sample)
        # prediction
        logits = fishr.classifier(features)
        predicted_label = logits.argmax(dim=1)[0].cpu().detach().numpy()
        _, grad_variance_augmented, _ = get_grad_average(fishr=fishr,
                                                      features=features,
                                                      loss_type=args.loss_type)
        # calculate the distance between the variance of the augmented dataset and the unaugmented dataset

        grad_variance_dist_mean = l2_between_dicts(grad_variance_augmented, grad_variance_mean)
        grad_variance_dist_per_class = l2_between_dicts(grad_variance_augmented,
                                                        grad_variance_per_class[predicted_label.item()])

        logger.log_custom("dist_to_mean", grad_variance_dist_mean.cpu().detach().numpy())
        logger.log_custom("dist_to_class", grad_variance_dist_per_class.cpu().detach().numpy())
        logger.log_custom("gt_label", gt_label[0].cpu().detach().numpy())
        logger.log_custom("predicted_label", predicted_label)
        logger.add_record_to_log_file()

        temp += 1
        # print temp every 100 samples
        if temp % 100 == 0:
            print("Step:{}, Time Elapsed:{}".format(temp, time.time() - t1))


def prediction_idea_4(args):
    logger = Logger(args.log_dir, "logs_{}_{}_{}_{}.csv".format(args.dname, args.model_name,
                                                                args.loss_type, args.corruption))

    fishr = Model(model_name=args.model_name, model_dir=args.model_dir, device=device)

    # Load the target dataset
    target = get_dataset(args.dname, args.data_dir, args.corruption, corr_level=4)
    # target = torch.utils.data.Subset(target, list(range(200)))
    # target.num_classes = 10
    # print("Target Dataset size", len(target))
    dataloader_target = DataLoader(target, batch_size=args.batch_size,
                                   shuffle=True, num_workers=10)
    acc, features, labels, predictions, logits, sampled_labels = fishr.get_acc_and_features(dataloader_target)
    print("Accuracy", acc)

    # Get the gradient with respect to the most probable prediction
    # fishr.classifier = fishr.classifier.to(device)
    fishr.classifier.train()
    features = features.to(device)

    temp = 0

    for sample, gt_label in zip(features, labels):

        # zero the gradients
        fishr.classifier.zero_grad()

        # print(sample.shape)
        # clone the sample 10 times
        sample = sample.repeat(10, 1)
        # print(sample.shape)
        # for each sample, change the label to be one of the other labels
        all_labels = torch.arange(10).to(device)
        # print(labels)

        logits = fishr.classifier(sample)

        predicted_label = logits.argmax(dim=1)[0].cpu().detach().numpy()

        # calculate the loss with respect to this label
        loss = fishr.bce_extended(logits, all_labels).sum()

        # calculate the gradient
        gc.collect()
        with backpack(BatchGrad()):  # , Variance()
            loss.backward()
        grad_mean, grad_var, _ = fishr.get_grad()

        grad_mean_over_var = OrderedDict()
        # for all the parameters of the layer (aka weight and bias)
        for key in grad_mean.keys():
            grad_mean_over_var[key] = grad_mean[key] / grad_var[key]

        logger.log_custom("grad_mean", norm_of_the_dict(grad_mean).cpu().detach().numpy())
        logger.log_custom("grad_var", norm_of_the_dict(grad_var).cpu().detach().numpy())
        logger.log_custom("grad_mean_over_var", norm_of_the_dict(grad_mean_over_var).cpu().detach().numpy())
        logger.log_custom("gt_label", gt_label.cpu().detach().numpy())
        logger.log_custom("predicted_label", predicted_label)
        logger.log_custom("if_prediction_is_correct", gt_label == predicted_label)
        logger.add_record_to_log_file()

        temp += 1
        # print temp every 100 samples
        if temp % 100 == 0:
            print(temp)
    print("Done")
    return



if __name__ == '__main__':
    seed_everything(42)
    args = get_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device", device)
    prediction_idea_2_full_net(args)
    """
    for corruption in [# "gaussian_noise",
                      "shot_noise", "impulse_noise",
                       # "defocus_blur",
                       "glass_blur", "motion_blur",
                       "zoom_blur", "snow",
                       "frost", "fog",
                       "brightness", "contrast",
                       "elastic_transform", "pixelate", "jpeg_compression"
        ]:
        args.corruption = corruption
    """
