
import copy
import os
import torch
from torch.utils.data import DataLoader
from data.datasets import get_dataset
import numpy as np

from utils.get_args import get_args, seed_everything
from utils.logger import Logger
from main import Model

from utils.utils import cov_pytorch, get_grad, get_grad_norm_pytorch
from utils.multivar_distr import log_likelihood_multivar, log_prob_logsumexp_trick, logits_to_prob_multivar, \
    batch_mahalanobis, get_per_class_log_probs
from wilds.common.data_loaders import get_eval_loader


def compute_gmm_stats(logits_per_class, num_classes, device="cuda:0"):
    # Calculate the mean and covariance matrix for each class
    # Input: logits_per_class - dictionary with keys as class labels and values as lists of logits

    gmm_stats = {}
    mean_per_class, cov_per_class, count_per_class = {}, {}, {}

    # Combine all the logits from the dictionary into one array

    all_logits = [logit for i in list(logits_per_class.keys()) for logit in logits_per_class[i]]
    all_logits = torch.stack(all_logits)

    # Normalize all logits and calculate the covariance matrix
    all_maxs, all_mins = torch.max(all_logits, dim=0)[0], torch.min(all_logits, dim=0)[0]
    all_logits = (all_logits - all_mins) / (all_maxs - all_mins)

    # mean of all logits
    all_means = torch.mean(all_logits, dim=0)
    cov_matr = cov_pytorch(all_logits, rowvar=False, bias=False)

    # Add small value to the diagonal to avoid division by zero
    cov_matr = cov_matr + torch.eye(cov_matr.shape[0], device=device) * 1e-5
    # use pseudoinverse to get the inverse of the covariance matrix
    # cov_matr = torch.pinverse(cov_matr)

    scales = []

    for cls in range(num_classes):
        if cls in logits_per_class.keys():
            count_per_class[cls] = len(logits_per_class[cls])
            if len(logits_per_class[cls]) > 5:
                logits_per_class[cls] = torch.stack(logits_per_class[cls])
                logits_per_class[cls] = (logits_per_class[cls] - all_mins) / (all_maxs - all_mins)
                mean_per_class[cls] = torch.mean(logits_per_class[cls], dim=0)
                # multivar_normal[cls] = torch.distributions.MultivariateNormal(mean_per_class[cls],
                #                                                              cov_matr)
            else:
                # make a covariance matrix of ones
                # print("Entering this else")
                # mean_per_class[cls] = None
                mean_per_class[cls] = torch.zeros(all_logits.shape[1], device=device)
                # mean_per_class[cls] = torch.mean(all_logits, dim=0)
        else:
            # make a covariance matrix of ones
            print("Entering this other else")
            mean_per_class[cls] = torch.zeros(all_logits.shape[1], device=device)
            # mean_per_class[cls] = torch.mean(all_logits, dim=0)
            # mean_per_class[cls] = None

            # multivar_normal[cls] = torch.distributions.MultivariateNormal(mean_per_class[cls], cov_matr)
    # Save means per class in a file
    # means_to_save = mean_per_class.values()
    # means_to_save = torch.stack(list(means_to_save)).cpu().numpy()
    # cov_to_save = cov_matr.cpu().numpy()

    gmm_stats["mean_per_class"] = mean_per_class
    gmm_stats["cov"] = cov_matr
    gmm_stats["scales"] = scales
    gmm_stats["maxs"] = all_maxs
    gmm_stats["mins"] = all_mins
    gmm_stats["count_per_class"] = count_per_class
    return gmm_stats



from scipy.stats import multivariate_normal
def manual_weight_estimation(X, means, covariance):
    n_components = len(means)
    n_samples = X.shape[0]

    # Initialize weights uniformly
    weights = np.ones(n_components) / n_components

    # E-step: Compute responsibilities
    responsibilities = np.zeros((n_samples, n_components))
    for i, mean in enumerate(means):
        rv = multivariate_normal(mean, covariance)
        responsibilities[:, i] = rv.pdf(X)
        print(responsibilities[:, i])
    responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)

    # M-step: Update weights
    weights = responsibilities.mean(axis=0)
    return weights



def get_per_class_mean_and_cov(dataloader, fishr, num_classes, use_labels=False):

    # Pass all samples through the network and store all the logits
    # Calculate the mean and covariance matrix for each class
    # Based on these means and covariances - calculate the multivariate normal distributions for each class


    logits_per_class = {cls: [] for cls in range(num_classes)}

    all_logits = []
    for batch_number, data in enumerate(dataloader):
        features, lbl = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            # features = fishr.feature_extractor(im)
            # Cauchy activation
            # features = cauchy_activation(features)
            logits_without_temp = fishr.classifier(features)
            all_logits.append(logits_without_temp.detach().cpu().numpy())
            predicted_classes = torch.argmax(logits_without_temp, dim=1)

            if use_labels:
                for i, label in enumerate(lbl):
                    logits_per_class[label.item()].append(logits_without_temp[i])
            else:
                for i, pred_class in enumerate(predicted_classes):
                    logits_per_class[pred_class.item()].append(logits_without_temp[i])  # .detach().cpu().numpy())

        if batch_number % 20 == 0:
            print("Batch:{} Out of:{}".format(batch_number, len(dataloader)))
            # break

    gmm_stats = compute_gmm_stats(logits_per_class, num_classes=num_classes)

    known_means = list(gmm_stats["mean_per_class"].values())
    known_means = torch.stack(known_means).cpu().numpy()
    all_logits = np.concatenate(all_logits, axis=0)
    # from sklearn.mixture import BayesianGaussianMixture
    # dpgmm = BayesianGaussianMixture(n_components=len(known_means),
    #                                 covariance_type='full',
    #                                 weight_concentration_prior_type='dirichlet_process',
    #                                 mean_precision_prior=1.0,  # Adjust this as necessary
    #                                 init_params='random',
    #                                 max_iter=1000)
    # print("Prior to fitting logits")
    # dpgmm.fit(all_logits)
    # weights = dpgmm.weights_
    # print(weights)
    # gmm_stats["weights"] = weights

    # for i in range(len(dpgmm.means_)):
    #     gmm_stats["mean_per_class"][i] = torch.Tensor(dpgmm.means_[i]).to(device)


    # np.save("./logs/means_per_class.npy", means_to_save)
    # np.save("./logs/cov_per_class.npy", cov_to_save)
    return gmm_stats


def forward_and_get_probab(classifier, features, num_classes, gmm_stats=None, per_class_logs=None):
    classifier.zero_grad()
    logits_without_temp = classifier(features)
    logits = copy.deepcopy(logits_without_temp.detach())
    if gmm_stats is not None:
        logits_without_temp = (logits_without_temp - gmm_stats['mins']) / (gmm_stats['maxs'] - gmm_stats['mins'])
        prob = log_prob_logsumexp_trick(gmm_stats, per_class_logs, logits_without_temp, num_classes,
                                        L2_regularizer=args.l2_reg,#  count_per_class=gmm_stats['count_per_class']
                                        )
    else:
        # Softmax to get the probabilities
        prob = torch.softmax(logits_without_temp, dim=1)
    return prob, logits, logits_without_temp


def load_features(dataloader, feature_extractor, fname):
    feature_extractor.eval()
    feature_extractor.to(device)
    if not os.path.exists(fname):
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

    print("Loading the features from the file")
    existing_embeddings_np = np.loadtxt(fname, delimiter=',')
    existing_embeddings_torch = torch.Tensor(existing_embeddings_np).type(torch.FloatTensor).to(device)
    existing_labels = np.loadtxt(fname.replace('.csv', '_label.csv'), delimiter=',')
    existing_labels = torch.Tensor(existing_labels).type(torch.LongTensor).to(device)
    # Dataloader to iterate through the embeddings and labels
    dset = torch.utils.data.TensorDataset(existing_embeddings_torch, existing_labels)

    return dset



def get_grads_and_store(fishr, feature_loader, logger, num_classes=10,
                        dataloader_source=None, use_gmm=True):
    # Move the model to the device and set it to eval mode
    fishr.feature_extractor.eval()
    fishr.feature_extractor.to(device)
    fishr.classifier.eval()
    fishr.classifier.to(device)

    print(f"Num classes: {num_classes}")
    if use_gmm:
        # Get and store the means and variances from the source dataset
        if dataloader_source is not None:
            gmm_stats = get_per_class_mean_and_cov(dataloader_source, fishr, num_classes=num_classes,
                                                   use_labels=True)
        else:
            gmm_stats = get_per_class_mean_and_cov(feature_loader, fishr, num_classes=num_classes)
        per_class_logs = get_per_class_log_probs(gmm_stats['mean_per_class'], gmm_stats['cov'],
                                                 num_classes, L2_regularizer=args.l2_reg)
    else:
        gmm_stats, per_class_logs = None, None

    for data in feature_loader:
        features, lbl = data[0].to(device), data[1].to(device)
        # with torch.no_grad():
        #    features = fishr.feature_extractor(im)
            # Cauchy activation
            # features = cauchy_activation(features)

        prob, logits, _ = forward_and_get_probab(fishr.classifier, features, num_classes, gmm_stats, per_class_logs)
        pred_class = torch.argmax(logits, dim=1).detach().cpu()
        #uniform_dist = torch.ones_like(prob) / num_classes
        #dele = get_grad_norm_pytorch(fishr, prob, uniform_dist)
        grad_weight_to_rand = {}
        grad_weight_to_target = {}
        temperature_range = np.arange(0.1, 1.3, 0.1)

        if not use_gmm:
            for temperature in temperature_range:

                prob, _, logits_attached = forward_and_get_probab(fishr.classifier, features, num_classes, gmm_stats,
                                                      per_class_logs)
                logits = logits_attached * temperature
                prob = torch.softmax(logits, dim=1)
                uniform_dist = torch.ones_like(prob) / num_classes
                loss_random, _, grad_weight_to_rand[temperature], grad_weight_to_rand_bias = get_grad_norm_pytorch(fishr, prob,
                                                                                                      uniform_dist)

                prob, _, logits_attached = forward_and_get_probab(fishr.classifier, features, num_classes, gmm_stats,
                                                      per_class_logs)
                logits = logits_attached * temperature
                prob = torch.softmax(logits, dim=1)
                pseudo_labels = torch.argmax(logits, dim=1)
                one_hot = torch.nn.functional.one_hot(pseudo_labels, num_classes=num_classes)
                # Log one_hot
                loss_target, _, grad_weight_to_target[temperature], grad_weight_to_target_bias = get_grad_norm_pytorch(fishr, prob,
                                                                                                          one_hot)

            # ========   Logs   =========
            # indx - index of a sample in a batch. Log each sample separately
            for indx, label in enumerate(lbl):
                logger.log_custom("labels", label.detach().cpu().item())
                logger.log_custom("pred", pred_class[indx].detach().cpu().item())
                for temperature in temperature_range:
                    logger.log_custom("temp{}_weight2rand".format(round(temperature, 1)),
                                      grad_weight_to_rand[temperature][indx].detach().cpu().item())
                    logger.log_custom("temp{}_weight2target".format(round(temperature, 1)),
                                      grad_weight_to_target[temperature][indx].detach().cpu().item())

                logger.add_record_to_log_file()
        else:
            uniform_dist = torch.ones_like(prob) / num_classes
            loss_random, _, grad_weight_to_rand, grad_weight_to_rand_bias = get_grad_norm_pytorch(fishr, prob, uniform_dist)

            prob, logits = forward_and_get_probab(fishr.classifier, features, num_classes, gmm_stats, per_class_logs)
            pseudo_labels = torch.argmax(logits, dim=1)
            one_hot = torch.nn.functional.one_hot(pseudo_labels, num_classes=num_classes)
            # Log one_hot
            loss_target, _, grad_weight_to_target, grad_weight_to_target_bias = get_grad_norm_pytorch(fishr, prob, one_hot)

            pred_class = torch.argmax(logits, dim=1).detach().cpu()
            prediction_prob, predicted_class = torch.max(prob, dim=1)

            # ========   Logs   =========
            # indx - index of a sample in a batch. Log each sample separately
            for indx, label in enumerate(lbl):
                logger.log_custom("labels", label.detach().cpu().item())
                logger.log_custom("pred", pred_class[indx].detach().cpu().item())
                logger.log_custom("pred_updated", predicted_class[indx].detach().cpu().item())

                logger.log_custom("loss_target", loss_target[indx].detach().cpu().item())
                logger.log_custom("loss_random", loss_random[indx].detach().cpu().item())
                logger.log_custom("weight2rand", grad_weight_to_rand[indx].detach().cpu().item())
                logger.log_custom("weight2target", grad_weight_to_target[indx].detach().cpu().item())

                logger.log_custom("weight2rand_bias", grad_weight_to_rand_bias[indx].detach().cpu().item())
                logger.log_custom("weight2target_bias", grad_weight_to_target_bias[indx].detach().cpu().item())
                # for k in range(1, 11):
                #     logger.log_custom("weight2k_{}".format(k), grad_weight_to_k[k][indx].detach().cpu().item())
                logger.add_record_to_log_file()

    print("FINISHED")
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


    target_dset, source_dset, num_classes = get_dataset(args.dname, args.data_dir, args.corruption,
                                                        use_ood_val=args.use_ood_val)

    for t_dset_name in target_dset.keys():
        t_set = target_dset[t_dset_name]
        if "camelyon" in args.dname:
            dataloader_target = get_eval_loader("standard", t_set, batch_size=args.batch_size)
            dataloader_source = get_eval_loader("standard", source_dset, batch_size=args.batch_size)
        else:
            dataloader_target = DataLoader(t_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
            dataloader_source = DataLoader(source_dset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        fishr = Model(model_name=args.model_name, model_dir=args.model_dir, device=device, num_classes=num_classes)

        log_dir = os.path.join(args.log_dir, args.dname)

        print(f"{log_dir}/features/features_{t_dset_name}_{args.corruption}.csv")
        feature_set = load_features(dataloader_target, fishr.feature_extractor,
                                    f"{log_dir}/features/features_{t_dset_name}_{args.corruption}.csv")
        feature_loader = DataLoader(feature_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

        if not args.use_gmm:
            logger_metrics = Logger(log_dir + "/fixed_temp/", f"gnorm_{t_dset_name}_{args.corruption}.csv")

            _ = get_grads_and_store(fishr, feature_loader, logger_metrics, num_classes=num_classes, use_gmm=False)
            # return True
        else:
            if args.use_source:
                logger_metrics = Logger(log_dir, f"gnorm_gmm_source_{t_dset_name}_{args.corruption}.csv")
                _ = get_grads_and_store(fishr, feature_loader,
                                    dataloader_source=dataloader_source, logger=logger_metrics, num_classes=num_classes)
            else:
                logger_metrics = Logger(log_dir, f"gnorm_gmm_target_{t_dset_name}_{args.corruption}.csv")
                _ = get_grads_and_store(fishr, feature_loader, logger=logger_metrics, num_classes=num_classes)

    return True


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device", device)
    args = get_args()
    seed_everything(args.rand_seed)
    # Store the results

    calculate_stats(args)