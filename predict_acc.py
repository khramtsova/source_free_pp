# read csv file
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import numpy as np
import copy
# read csv file
# read all files in the folder and concatenate them into one dataframe
from utils.plot_utils import plot_boxplot_reach_gmm, plot_scatter_openness
#
# # GNorm


def get_gnorm_results(path_to_csv):
    maes, accs_gt = [], []
    csv_files = glob.glob(path_to_csv + "/*.csv")
    sorted_csv_files = sorted(csv_files)
    for file in  sorted_csv_files:
        df = pd.read_csv(file)
        acc_gt = sum(df['labels'] == df['pred'])*100
        prediction = sum(df['weight2rand']>=df['weight2target'])*100
        maes.append(abs(acc_gt/len(df) - prediction/len(df)))
        accs_gt.append(acc_gt/len(df))
    return np.mean(accs_gt), np.mean(maes), accs_gt, maes

def get_smallest_index_per_corr(mae_per_corr_full, gnorm_mae, open_range,
                                feature, average_across_corr=True):
    # find the first True value
    mae_per_corr = mae_per_corr_full[feature]
    corr_range = len(mae_per_corr[openness_range[0]])

    if average_across_corr:
        first_smallest_index = 1.0
    else:
        first_smallest_index = [1.0 for _ in range(corr_range)]
    for open in open_range:
        if average_across_corr:
            smaller = np.mean(mae_per_corr[open]) < np.mean(gnorm_mae)
            if smaller:
                print("HEREV2", open, np.mean(mae_per_corr[open]), np.mean(gnorm_mae))
                return open
        else:
            smaller = mae_per_corr[open] < gnorm_mae
            for i in range(corr_range):
                if smaller[i] and first_smallest_index[i] == 1.0:
                    first_smallest_index[i] = open

    return first_smallest_index
#
# path = 'logs/logs_v2/gnorm_gmm/zeros_no_prior/imagenet/'
# path = 'logs/DEL1/iwildcam/'
# maes, accs_gt = [], []
# # iterate through all files in the folder
# for file in glob.glob(path + "/*.csv"):
#     df = pd.read_csv(file)
#     acc_gt = sum(df['labels'] == df['pred'])*100
#     prediction = sum(df['weight2rand']>=df['weight2target'])*100
#     maes.append(abs(round(acc_gt/len(df) - prediction/len(df), 2)))
#     accs_gt.append(acc_gt/len(df))
#     print(f"File name: {file}")
#     print(f"Accuracy: {acc_gt/len(df)}, Prediction: {prediction/len(df)}")
#     print(f"MAE: {abs(round(acc_gt/len(df) - prediction/len(df), 2))}")
# print(f"Accuracy: {round(np.mean(accs_gt), 2)}")
# print(f"MAE total: {round(np.mean(maes), 2)}")

#
# path = './logs/logs_v2/other_baselines/terra_incognita/'
#
# # iterate through all files in the folder
# # and concatenate them into one dataframe
# df = pd.concat([pd.read_csv(file) for file in glob.glob(path + "/*.csv")], ignore_index=True)
# # df = pd.read_csv(path + "logs_acc_based_on_metrics_v2.csv")
#
# for column in df.columns:
#     if "mae" not in column:
#         continue
#     values = df[column].values
#     print(f"{column}: {round(values.mean()*100, 2)}")

#
# dset = "pacs"
# # dset = "camelyon"
# # feature = "mae_energy_based_calibr"
# feature = "mae_atc_entr"
# # feature = "mae_cot_based"
#
# res = {}
# path_orig = f'./logs/logs_v2/other_baselines/{dset}/'
# gt_acc, gnorm_gmm_mae, _, _ = get_gnorm_results(f'./logs/logs_v2/gnorm_gmm/means/{dset}/')
#
# res["GMM + GNorm"] = gnorm_gmm_mae
#
# for openness in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,  0.7, 0.8, 0.9]:
#     path = os.path.join(path_orig, f"openness_{openness}")
#
#     for seed in [0,1,2,3]: #, 42]:
#         path_to_rand = os.path.join(path_orig, f"openness_{openness}", f"seed_{seed}")
#         df = pd.concat([pd.read_csv(file) for file in glob.glob(path_to_rand + "/*.csv",
#                                                                 recursive=True)],
#                    ignore_index=True)
#         if openness not in res.keys():
#             res[openness] = []
#         res[openness].append(df[feature].values.mean()*100)
#
#     # res[openness] = df[feature].values*100
# # for openness in res.keys():
# #     print(f"Openness: {openness}, {np.mean(res[openness])}")
#
# plt.boxplot(res.values(), labels=res.keys())
# plt.xlabel("Openness")
# plt.ylabel("MAE")
# feature_name = feature.split("_")[1]
# plt.title(f"{feature_name} for {dset} dataset")
# plt.show()
#


import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dset = "vlcs" # Example dataset
features = ["mae_atc_entr",
            "mae_atc_prob",
            "mae_doc_based",
            "mae_cot_based",
            "mae_energy_based",
            # "mae_nuc_based_calibr"
            ]  # Example feature

#
# features = ["mae_atc_entr_calibr",
#             "mae_atc_prob_calibr",
#             "mae_doc_based_calibr",
#             "mae_cot_based_calibr",
#             "mae_energy_based_calibr",
#             ]

path_orig = f'./logs/logs_v2/other_baselines/{dset}/'
# Assuming get_gnorm_results is a function you have defined elsewhere
(gt_acc, gnorm_gmm_mae,
 gt_acc_per_dset, gnorm_gmm_mae_per_dset) = get_gnorm_results(f'./logs/logs_v2/gnorm_gmm/zeros/{dset}/')

print(gt_acc_per_dset)

dset_names = {
    "cifar10": "CIFAR-10.1", "fmow": "FMoW",
    "iwildcam": "iWildCam", "rxrx1": "RxRx1", "camelyon": "Camelyon",
    "vlcs": "VLCS", "terra_incognita": "TerraIncognita",
    "digits": "Digits", "pacs": "PACS",  "domain_net": "DomainNet", "office_home": "OfficeHome",
}
methods_names = {
    "mae_ac": "AC", "mae_nuc_based": "Nuclear Norm",
    "mae_atc_entr": "ATC-Entropy", "mae_atc_prob": "ATC-Probability",
    "mae_cot_based": "COT", "mae_energy_based": "Energy",
    "mae_doc_based": "DOC"
}


# Modified part to accumulate MAE values and their variances
mae_values, variances = {}, {}


seed_range = list(range(20))  # Range of seeds
seed_range.append(42)

index_to_reach_gnorm = {f"corr_{i}": -1 for i in range(len(gnorm_gmm_mae_per_dset))}
indices_to_gnorm = {feature: copy.copy(index_to_reach_gnorm) for feature in features}



#print("indices_to_gnorm Prior", indices_to_gnorm )

openness_range = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
percentages = [1, 5, 10, 20 , 30, 40, 50, 60, 70, 80, 90, 100]  # Representing openness as percentages

exact_mae_per_corr, exact_var_per_corr = {}, {}
for feature in features:
    exact_mae_per_corr[feature] = {}
    exact_var_per_corr[feature] = {}
    for openness in openness_range:
        exact_mae_per_corr[feature] [openness]= []
        exact_var_per_corr[feature] [openness]= []

for openness in openness_range: #, 1.0]:
    mae_for_openness = [[] for _ in features]  # [[mae_atc_entr], [mae_cot]]
    if openness == 1.0:
        seed_range = [42]
    for seed in seed_range:  # Loop over seeds
        path_to_rand = os.path.join(path_orig, f"openness_{openness}", f"seed_{seed}")
        # print(path_to_rand)
        df = pd.concat([pd.read_csv(file) for file in glob.glob(path_to_rand + "/*.csv",
                                                                recursive=True)],
                       ignore_index=True)
        try:
            assert len(df) == len(gnorm_gmm_mae_per_dset)
        except:
            print(path_to_rand, len(df), len(gnorm_gmm_mae_per_dset))
            raise
        for i, feature in enumerate(features):
            mae_for_openness[i].append(df[feature].values * 100)
    # Calculate mean and variance for each openness
    for feature_indx, feature in enumerate(features):
        if feature not in mae_values.keys():
            mae_values[feature] = []
            variances[feature] = []

        mean_across_tests = np.mean(mae_for_openness[feature_indx], axis=1)
        mae_values[feature].append(np.mean(mean_across_tests))
        variances[feature].append(np.var(mean_across_tests))

        mean_across_seeds = np.mean(mae_for_openness[feature_indx], axis=0)
        var_across_seeds = np.var(mae_for_openness[feature_indx], axis=0)
        exact_mae_per_corr[feature][openness] = mean_across_seeds
        exact_var_per_corr[feature][openness] = var_across_seeds


# for feature in features:
#     for openness in [0.01]:
#        for mae, var in zip(exact_mae_per_corr[feature][openness], exact_var_per_corr[feature][openness]):
#            print(feature, openness, round(mae,2), round(np.sqrt(var),2))
#        print(feature, openness, exact_mae_per_corr[feature][openness])
#     print(feature, round(mae_values[feature][0], 2),
#           round( np.sqrt(variances[feature][0]), 2 ))
# print(exact_mae_per_corr["mae_nuc_based_calibr"])
#raise
# print("Our Method", gnorm_gmm_mae_per_dset)
# print(exact_mae_per_corr["mae_cot_based"])

smallest_indx = {}
for feature in features:
    smallest_indx[feature] = get_smallest_index_per_corr(exact_mae_per_corr,
                                            gnorm_gmm_mae_per_dset,
                                            openness_range,
                                            feature=feature,
                                            average_across_corr=False)


    # print(exact_mae_per_corr[feature][smallest_indx],
    #       np.mean(exact_mae_per_corr[feature][smallest_indx]),
    #       gnorm_gmm_mae)
print(smallest_indx)
# with open(f'./logs/logs_v2/percentage_to_reach_gmm/percentage_to_reach_gnorm_{dset}.csv', 'w') as f:
#     # save nested dictionary into a csv file
#     pd.DataFrame(smallest_indx).to_csv(f, header=True, index=False)
# plot_boxplot_reach_gmm(smallest_indx, dset, dset_names[dset], methods_names, fontsize=25)
# plt.tight_layout()
# plt.savefig(f'./logs/plots/boxplots/{dset}_comparison.png', dpi=300, bbox_inches='tight')
# plt.show()
print("HERE")
print(features)
print(percentages)
print(mae_values)
print(variances)
print(gnorm_gmm_mae)
print(dset)
print(methods_names)
print(dset_names)
print(df)


plot_scatter_openness(features, percentages, mae_values, variances, gnorm_gmm_mae,
                        dset, methods_names, dset_names, df=df)
plt.tight_layout()
# plt.savefig(f'./logs/plots/{dset}_full.png', dpi=300, bbox_inches='tight')
plt.show()


# print(mae_values)
# for feature in features:
#     print(feature, round(mae_values[feature][0], 2),
#           round( np.sqrt(variances[feature][0]), 2 ))
# raise
# # Print features for 1% of training data
#
#
# print(mae_values[feature])
# print(gnorm_gmm_mae)
# raise
# percentage_to_reach_gnorm = {}
# for feature in features:
#     # find the first True value
#     smaller = mae_values[feature] < gnorm_gmm_mae
#     print(feature, smaller)
#     print(gnorm_gmm_mae, mae_values[feature])
#     if not any(smaller):
#         first_smallest_index = -1
#     else:
#         first_smallest_index = np.argmax(mae_values[feature] < gnorm_gmm_mae)
#     percentage_to_reach_gnorm[feature] = percentages[first_smallest_index]
# # save into a csv file
# with open(f'./logs/logs_v2/percentage_to_reach_gmm/percentage_to_reach_gnorm_{dset}.csv', 'w') as f:
#     pd.DataFrame(percentage_to_reach_gnorm, index=[0]).to_csv(f, header=True, index=False)
# print(percentage_to_reach_gnorm)
#
# raise

#
#
# # Plots
# plt.style.use('seaborn-v0_8-poster')  # ECCV-style plot
# plt.figure(figsize=(10, 8))
# source_free_methods_shapes = {'mae_ac': '>', 'mae_nuc_based': '<'}
# # for source_free_methods in ["mae_ac", "mae_nuc_based"]:
# for feature in features:
#     for i, (perc, mae, var) in enumerate(zip(percentages, mae_values[feature], variances[feature])):
#         plt.vlines(perc, mae - np.sqrt(var), mae + np.sqrt(var), color='grey', alpha=0.5)
#         #plt.errorbar(perc, mae, yerr=np.sqrt(var), color='grey',   alpha=0.8,
#         #             capsize=5, capthick=2, elinewidth=2,
#         #             )
#
# print(plt.style.available)
# line_styles = ['--','-',  '-.', ':']  # Custom dash pattern added
# marker_shapes = ['o', 's', '^',  'p', 'D',]  # Adding 'p' (pentagon) as a new marker shape
#
#
# for i, feature in enumerate(features):
#     line_style = line_styles[i % len(line_styles)]
#     marker_shape = marker_shapes[i % len(marker_shapes)]
#     plt.plot(percentages, mae_values[feature],# '-o',
#              line_style, label=methods_names[feature], linewidth=4, markersize=13, marker=marker_shape)  # Connect points with a line
#     # print(i)
#     #plt.plot(percentages, mae_values, '-o', label=feature, linewidth=2, markersize=8, )  # Connect points with a line
#
# for method, shape in source_free_methods_shapes.items():
#     values = df[method].mean() * 100
#     plt.scatter(0, values, label=methods_names[method], zorder=5, s=200, marker=shape)
#
#
#
# fontsize = 22
# plt.scatter(0, gnorm_gmm_mae, label='Our Method',  zorder=5, s=300, marker="X", c="black")  # First point style
#
# plt.xlabel('Openness (%)', fontsize=fontsize+8, fontweight='bold')
# plt.ylabel('MAE (%)', fontsize=fontsize+8, fontweight='bold')
# plt.title(f'{dset_names[dset]}', fontsize=fontsize+8, fontweight='bold')
#
# plt.xticks(ticks=[0, 1,5, 10, 20],fontsize=fontsize+6,fontweight='bold')
# plt.yticks(fontsize=fontsize+6, fontweight='bold')
# # position: top right, transparent background
# # plt.legend(fontsize=fontsize, loc='upper right', framealpha=0.93 )
# plt.grid(True, alpha=0.2)
# # plt.savefig(f'./logs/plots/{dset}.png', dpi=300, bbox_inches='tight')
# plt.show()