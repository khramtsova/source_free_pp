import os.path

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn
from data.datasets import get_dataset
from utils.multivar_distr import log_prob_logsumexp_trick

from pred_source_free import compute_gmm_stats, get_per_class_log_probs
from utils.get_args import get_args, seed_everything
from utils.logger import Logger
from main import Model
from utils.ATC_helpers.utils import softmax_numpy, inverse_softmax_numpy, get_entropy
from sklearn.manifold import TSNE

import ot
import torch.nn.functional as F



def set_bn_to_train(net):
    # function that iterates through the layers of the network
    # and change all the batch norm layers to train mode\
    for layer in net.modules():
        if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
            layer.train()


def get_logits_and_preds(fishr, dset, fname=None, device="cuda", subset_perc=1.0):
    # Move the model to the device and set it to eval mode
    fishr.feature_extractor.eval()
    fishr.feature_extractor.to(device)
    fishr.classifier.eval()
    fishr.classifier.to(device)
        #set_bn_to_train(fishr.feature_extractor)
    #set_bn_to_train(fishr.classifier)

    # get and store the prediction probabilities for the source dataset
    probs, labels, loss_logsoftmax = [], [], []

    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    logits_list, features_list = [], []

    # read the embeddings from the file
    # existing_embeddings_np = np.loadtxt('embeddings.csv', delimiter=',')
    if fname is not None:
        if not os.path.exists(fname):
            # create a folder and a file to store the embeddings
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            f = open(fname, 'a')
            f_label = open(fname.replace('.csv', '_label.csv'), 'a')
            emb_exists = False
        else:
            # if the file exists, do not save anything
            emb_exists = True
    if emb_exists:
        # read the embeddings from the file
        existing_embeddings_np = np.loadtxt(fname, delimiter=',')
        existing_embeddings_torch = torch.Tensor(existing_embeddings_np).type(torch.FloatTensor).to(device)
        existing_labels = np.loadtxt(fname.replace('.csv', '_label.csv'), delimiter=',')
        existing_labels = torch.Tensor(existing_labels).type(torch.LongTensor).to(device)
        # Dataloader to iterate through the embeddings and labels
        print("Existing embeddings shape", existing_embeddings_torch.shape, "Existing labels shape", existing_labels.shape)
        dset = torch.utils.data.TensorDataset(existing_embeddings_torch, existing_labels)

    if subset_perc < 1.0:
        rand_index = torch.randperm(len(dset))
        rand_index = rand_index[:int(subset_perc * len(dset))]
        dset = torch.utils.data.Subset(dset, rand_index)

    dataloader = DataLoader(dset, batch_size=64, shuffle=False, num_workers=0)

    for batch_indx, data in enumerate(dataloader):
        with torch.no_grad():
            if not emb_exists:
                im, lbl = data[0], data[1]
                im, lbl = im.to(device), lbl.to(device)
                features = fishr.feature_extractor(im)
                np.savetxt(f, features.detach().cpu().numpy(), delimiter=',')
                np.savetxt(f_label, lbl.detach().cpu().numpy(), delimiter=',')
            else:
                features, lbl = data
                features, lbl = features.to(device), lbl.to(device)
            logits = fishr.classifier(features)
            logits_list.append(logits)
            labels.append(lbl)
    if not emb_exists:
        f.close()

    # probs = np.concatenate(probs, axis=0)
    # loss_logsoftmax = np.concatenate(loss_logsoftmax, axis=0)

    labels = torch.cat(labels, dim=0)
    logits_list = torch.cat(logits_list, dim=0)

    return probs, labels, loss_logsoftmax, logits_list


# def get_entropy_and_probs(probs, labels, calibration=False):
#     pred_idx = np.argmax(probs, axis=-1)
#     source_acc = np.mean(pred_idx == labels) * 100.
#     if not calibration:
#         pred_probab = np.max(probs, axis=-1)
#         pred_indx = np.argmax(probs, axis=-1)
#         entropy = get_entropy(probs)
#         return entropy, pred_indx, pred_probab
#     probs_calibrated = softmax_numpy(inverse_softmax_numpy(probs))
#     pred_indx_calibrated = np.argmax(probs_calibrated, axis=-1)
#     pred_probab_calibrated = np.max(probs_calibrated, axis=-1)
#     calib_entropy = get_entropy(probs_calibrated)
#     return calib_entropy, pred_indx_calibrated, pred_probab_calibrated


def get_log_softmax(logits):
    ones = torch.ones(logits.shape)
    ones = ones.to(logits.device)
    loss_logsoftmax_source = torch.sum(-ones * F.log_softmax(logits), dim=1)
    return loss_logsoftmax_source


def cot(iid_tars, ood_acts, conf_gap):
    act = nn.Softmax(dim=1)
    iid_acts, ood_acts = nn.functional.one_hot(iid_tars), act(ood_acts)
    print("IID shape", iid_acts.shape, "OOD shape", ood_acts.shape)
    M = torch.from_numpy(ot.dist(iid_acts.cpu().numpy(), ood_acts.cpu().numpy(), metric='minkowski',
                                 p=1))  # .to(device)
    weights = torch.as_tensor([])  # .to(device)
    est = (ot.emd2(weights, weights, M, numItermax=10 ** 8) / 2 + conf_gap)
    return est


def doc(source_prob, source_labels, target_prob):
    # Difference of Confidences
    avrg_source_conf = source_prob.amax(1).mean().item()
    avrg_target_conf = target_prob.amax(1).mean().item()

    source_acc = (source_prob.argmax(1) == source_labels).float().mean().item()
    return source_acc + (avrg_target_conf - avrg_source_conf)


def nuclear_norm(target_probs):
    # transform to numpy
    target_scores = target_probs.cpu().detach()
    raw_nuclear = torch.norm(target_scores, p='nuc')
    # upper bound of nuclear norm
    n_classes = target_scores.shape[1]
    n_samples = target_scores.shape[0]
    n_norm = np.sqrt(n_samples * np.min([n_samples, n_classes]))
    # final score
    nuclear_score = raw_nuclear / n_norm
    return nuclear_score.item()


def ATC(source_prob, source_labels, target_prob):
    source_scores = source_prob.amax(1)
    target_scores = target_prob.amax(1)
    sorted_source_scores, _ = torch.sort(source_scores)
    threshold = sorted_source_scores[-(source_prob.argmax(1) == source_labels).sum()]
    estimate = (target_scores > threshold).float().mean().item()
    return estimate


def ATC_energy(source_logits, source_labels, target_logits):
    source_scores = torch.logsumexp(source_logits, dim=1)
    target_scores = torch.logsumexp(target_logits, dim=1)
    sorted_source_scores, _ = torch.sort(source_scores)
    threshold = sorted_source_scores[-(source_logits.argmax(1) == source_labels).sum()]
    estimate = (target_scores > threshold).float().mean().item()
    return estimate


def negentropy(probs):
    return (probs * torch.log(probs)).sum(1)


def ATC_negent(source_prob, source_labels, target_prob):
    source_scores = negentropy(source_prob)
    target_scores = negentropy(target_prob)
    sorted_source_scores, _ = torch.sort(source_scores)
    threshold = sorted_source_scores[-(source_prob.argmax(1) == source_labels).sum()]
    estimate = (target_scores > threshold).float().mean().item()
    return estimate


def exact_entropy(target_logits, threshold=0.1):
    num_classes = target_logits.shape[1]
    target_scores = - negentropy(target_logits)
    normalized_entropy = target_scores / torch.log(torch.tensor(num_classes))
    estimate = sum(normalized_entropy < threshold).float().cpu().item()/len(normalized_entropy)
    return estimate


# Confidence Optimal Transport
def COT(source_probs, source_labels, target_probs):
    num_classes = source_probs.shape[1]
    source_label_dist = torch.nn.functional.one_hot(source_labels, num_classes=num_classes).float().mean(0)
    cost_matrix = torch.stack([(target_probs - onehot).abs().sum(1)
                               for onehot in torch.eye(num_classes, device=source_probs.device)], dim=1) / 2
    ot_plan = ot.emd(torch.ones(len(target_probs)) / len(target_probs), source_label_dist, cost_matrix)
    ot_cost = torch.sum(ot_plan * cost_matrix.cpu()).item()

    # s_conf = torch.softmax(source_logits, dim=1).amax(1).mean().item()
    s_conf = source_probs.amax(1).mean().item()
    s_acc = (source_probs.argmax(1) == source_labels).float().mean().item()
    conf_gap = s_conf - s_acc
    err_est = ot_cost + conf_gap
    return 1. - err_est


# Average Confidence
def AC(target_prob):
    return target_prob.amax(1).mean().item()


def calibration_temp(logits, labels):
    log_temp = torch.nn.Parameter(torch.tensor([0.], device=logits.device))
    temp_opt = torch.optim.LBFGS([log_temp], lr=.1, max_iter=50, tolerance_change=5e-5)

    def closure_fn():
        loss = F.cross_entropy(logits * torch.exp(log_temp), labels)
        temp_opt.zero_grad()
        loss.backward()
        return loss

    temp_opt.step(closure_fn)
    return torch.exp(log_temp).item()


def get_logits_per_class(logits):
    logits_per_class = {}
    # consider the prediction to be the class with the highest probability
    prediction = logits.argmax(1)
    # make predictions to be a list instead of tensors
    prediction = prediction.cpu().detach().numpy()
    for prediction, logit in zip(prediction, logits):
        if prediction not in logits_per_class:
            logits_per_class[prediction] = []
        logits_per_class[prediction].append(logit)
    return logits_per_class


def make_tsne_plot(prob, lbl, num_classes):

    prob_tsne = TSNE(n_components=2, learning_rate='auto',
                     init='random', perplexity=11
                     ).fit_transform(prob.cpu())
    labels_target = lbl.cpu()
    prob_target = prob.cpu()
    import matplotlib.pyplot as plt
    for lbl in range(num_classes):
        subset = prob_tsne[labels_target == lbl]
        plt.scatter(subset.T[0], subset.T[1])
    plt.show()
    return True


def atc_method(args):

    log_dir = os.path.join(args.log_dir, args.dname)
    #if args.openness != 1.:
    #    print("Openness", args.openness, "Seed", args.rand_seed)
    log_dir = os.path.join(log_dir, "openness_{}".format(args.openness), f"seed_{args.rand_seed}")
    logger_acc = Logger(log_dir, "metric_based_{}.csv".format(args.corruption))

    if "domain_net" in args.dname:
        args.model_name = "resnet50_domainbed"
        # args.corruption is the index of corruption in this case
        # Example:  "./model/model_weights/resnet_domainbed/3/model_test_envs[3].pkl"
        args.model_dir = "{}/resnet50_domain_net/{}/model_test_envs[{}].pkl".format(args.model_dir,
                                                                                    args.corruption,
                                                                                    args.corruption)
    elif "pacs" in args.dname:
        args.model_name = "resnet50_domainbed"
        args.model_dir = "{}/resnet50_pacs/{}/model.pkl".format(args.model_dir, args.corruption)
    elif "vlcs" in args.dname:
        args.model_name = "resnet50_domainbed"
        args.model_dir = "{}/resnet50_vlcs/ERM/{}/model.pkl".format(args.model_dir, args.corruption)
    elif "office_home" in args.dname:
        args.model_name = "resnet50_domainbed"
        args.model_dir = "{}/resnet50_office_home/ERM/{}/model.pkl".format(args.model_dir, args.corruption)
    elif "terra_incognita" in args.dname:
        args.model_name = "resnet50_domainbed"
        args.model_dir = "{}/resnet50_terra_incognita/ERM/{}/model.pkl".format(args.model_dir, args.corruption)

    # source = get_dataset(args.dname_source, args.data_dir, corruption=args.corruption, corr_level=-1)
    # target = get_dataset(args.dname, args.data_dir, args.corruption, corr_level=4)

    target_dset, source, num_classes = get_dataset(args.dname, args.data_dir, args.corruption,
                                                   use_ood_val=args.use_ood_val)

    print("Prior to source logits")
    fishr = Model(model_name=args.model_name, model_dir=args.model_dir, device=device,
                  num_classes=num_classes)

    if args.openness != 1.:
        rand_index = torch.randperm(len(source))
        rand_index = rand_index[:int(args.openness * len(source))]
        source = torch.utils.data.Subset(source, rand_index)

    fname = os.path.join(args.log_dir, "embeddings", args.dname, f"{args.dname}_source{args.corruption}.csv")
    print("fname source", fname)

    _, labels_source, _, logits_source_uncalibrated = get_logits_and_preds(fishr, source, fname,
                                                                           subset_perc=args.openness)
    print("Source logits calculated")

    for t_dset_name in target_dset.keys():
        print("DSET NAME", t_dset_name, "Num classes", num_classes)
        pred_accuracies = {}
        target = target_dset[t_dset_name]

        fishr = Model(model_name=args.model_name, model_dir=args.model_dir, device=device,
                      num_classes=num_classes)
        # =================================================================================
        # Source-Related: get the predictions and the probabilities for the source dataset,
        # compute the threshold for the entropy and the threshold for the probabilities
        # =================================================================================

        # source = torch.utils.data.Subset(source, list(range(200)))
        # source.num_classes = 10

        fname = os.path.join(args.log_dir, "embeddings", args.dname, f"{t_dset_name}_{args.corruption}.csv")
        _, labels_target, _, logits_target_uncalibrated = get_logits_and_preds(fishr, target, fname)
        print("Target logits calculated")

        acc_GT = np.mean(logits_target_uncalibrated.argmax(1).cpu().detach().numpy() ==
                         labels_target.cpu().detach().numpy())

        source_temperature = calibration_temp(logits_source_uncalibrated.detach(),
                                              torch.tensor(labels_source))

        logger_acc.log_custom("corruption", t_dset_name)
        logger_acc.log_custom("GT_accuracy", acc_GT)
        for calibr_type in ["NoCal", "Calibr"]: # , "GMM_source", "GMM_target"]:
            if calibr_type == "NoCal":
                logits_source = logits_source_uncalibrated
                logits_target = logits_target_uncalibrated
                prob_source, prob_target = torch.softmax(logits_source, dim=1), torch.softmax(logits_target, dim=1)
                log_name = ""
            elif calibr_type == "Calibr":
                logits_source = logits_source_uncalibrated * source_temperature
                logits_target = logits_target_uncalibrated * source_temperature
                prob_source, prob_target = torch.softmax(logits_source, dim=1), torch.softmax(logits_target, dim=1)
                log_name = "_calibr"
            elif calibr_type == "GMM_source":
                logits_grouped_source = get_logits_per_class(logits_source_uncalibrated)
                gmm_stats_source = compute_gmm_stats(logits_grouped_source, num_classes=num_classes)
                per_class_logs_source = get_per_class_log_probs(gmm_stats_source['mean_per_class'], gmm_stats_source['cov'],
                                                                num_classes, L2_regularizer=args.l2_reg)

                logits_source = (logits_source_uncalibrated - gmm_stats_source['mins']) / (gmm_stats_source['maxs'] - gmm_stats_source['mins'])

                prob_source = log_prob_logsumexp_trick(gmm_stats_source, per_class_logs_source, logits_source,
                                                       num_classes, L2_regularizer=False)

                # prob_source_softmax, prob_target_softmax = (torch.softmax(logits_source_uncalibrated, dim=1),
                #                                             torch.softmax(logits_target_uncalibrated, dim=1))
                # make_tsne_plot(prob_target_softmax, labels_target, num_classes)

                logits_grouped = get_logits_per_class(logits_target_uncalibrated)
                gmm_stats = compute_gmm_stats(logits_grouped, num_classes=num_classes)
                per_class_logs = get_per_class_log_probs(gmm_stats['mean_per_class'], gmm_stats['cov'],
                                                         num_classes, L2_regularizer=args.l2_reg)
                logits_target = (logits_target_uncalibrated - gmm_stats['mins']) / (gmm_stats['maxs'] - gmm_stats['mins'])

                prob_target = log_prob_logsumexp_trick(gmm_stats, per_class_logs,
                                                       logits_target, num_classes, L2_regularizer=False)

                # make_tsne_plot(prob_target, labels_target, num_classes)
                # raise

                log_name = "_gmm_source"
            elif calibr_type == "GMM_target":
                logits_grouped = get_logits_per_class(logits_target_uncalibrated)
                gmm_stats = compute_gmm_stats(logits_grouped, num_classes=num_classes)
                per_class_logs = get_per_class_log_probs(gmm_stats['mean_per_class'], gmm_stats['cov'],
                                                         num_classes, L2_regularizer=args.l2_reg)
                logits_source = (logits_source_uncalibrated - gmm_stats['mins']) / (
                        gmm_stats['maxs'] - gmm_stats['mins'])
                logits_target = (logits_target_uncalibrated - gmm_stats['mins']) / (gmm_stats['maxs'] - gmm_stats['mins'])
                #
                prob_target = log_prob_logsumexp_trick(gmm_stats, per_class_logs,
                                                       logits_target, num_classes, L2_regularizer=False)
                prob_source = log_prob_logsumexp_trick(gmm_stats, per_class_logs,
                                                       logits_source, num_classes, L2_regularizer=False)
                log_name = "_gmm_target"

            # prob_source, prob_target = torch.softmax(logits_source, dim=1), torch.softmax(logits_target, dim=1)

            pred_accuracies["entropy01_based"] = exact_entropy(prob_target, threshold=0.1)
            pred_accuracies["entropy03_based"] = exact_entropy(prob_target, threshold=0.3)
            pred_accuracies["ac" + log_name] = AC(prob_target)
            pred_accuracies["atc_entr" + log_name] = ATC_negent(prob_source, labels_source, prob_target)
            pred_accuracies["atc_prob" + log_name] = ATC(prob_source, labels_source, prob_target)
            pred_accuracies["doc_based" + log_name] = doc(prob_source, labels_source, prob_target)
            pred_accuracies["cot_based" + log_name] = COT(prob_source, labels_source, prob_target)
            pred_accuracies["nuc_based" + log_name] = nuclear_norm(prob_target)

            pred_accuracies["energy_based" + log_name] = ATC_energy(logits_source, labels_source, logits_target)

        for key in pred_accuracies.keys():
            logger_acc.log_custom(key, pred_accuracies[key])
            mae = abs(pred_accuracies[key] - acc_GT)
            logger_acc.log_custom("mae_" + key, mae)
            print("Metric: {}, Accuracy: {}, MAE: {}".format(key, pred_accuracies[key],
                                                             round(100 * mae, 2)))
        logger_acc.add_record_to_log_file()

    return True


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device", device)
    args = get_args()
    seed_everything(args.rand_seed)
    atc_method(args)