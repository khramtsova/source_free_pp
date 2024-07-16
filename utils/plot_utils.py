

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_dset_corruption_names(dname):
    if "camelyon" in dname:
        return ["Camelyon"]
    elif "fmow" in dname:
        return ["FMoW"]
    elif "iwildcam" in dname:
        return ["iWildCam"]
    elif "pacs" in dname:
        return ["Art ", "Cartoon", "Photo ", "Sketch"]
    elif "domain_net" in dname:
        return ["Clipart ", "Infograph", "Painting", "Quickdraw", "Real", "Sketch"]
    elif "office_home" in dname:
        return ["Art", "Clipart", "Product", "Photo"]
    elif "vlcs" in dname:
        return ["Caltech", "Labelme", "Sun", "VOC2007"]
    elif "terra_incognita" in dname:
        return ["L100", "L38", "L43", "L46"]
    elif "digits" in dname:
        return ["SVHN","USPS", "Synth"]





def plot_scatter_openness(features, percentages, mae_values, variances, gnorm_gmm_mae,
                          dset, methods_names, dset_names, df):
    plt.style.use('seaborn-v0_8-poster')
    plt.figure(figsize=(10, 8))
    source_free_methods_shapes = {'mae_ac': '>', 'mae_nuc_based': '<'}
    # for source_free_methods in ["mae_ac", "mae_nuc_based"]:
    for feature in features:
        for i, (perc, mae, var) in enumerate(zip(percentages, mae_values[feature], variances[feature])):
            # plt.vlines(perc, mae - np.sqrt(var), mae + np.sqrt(var), color='grey', alpha=0.5)
            plt.errorbar(perc, mae, yerr=np.sqrt(var), color='grey',   alpha=0.8,
                         capsize=5, capthick=2, elinewidth=2,
                         )

    print(plt.style.available)
    line_styles = ['--', '--', '-', '-.', ':']  # Custom dash pattern added
    marker_shapes = ['o', 's', '^', 'p', 'D', ]  # Adding 'p' (pentagon) as a new marker shape

    for i, feature in enumerate(features):
        line_style = line_styles[i % len(line_styles)]
        marker_shape = marker_shapes[i % len(marker_shapes)]
        plt.plot(percentages, mae_values[feature],  # '-o',
                 line_style, label=methods_names[feature], linewidth=4, markersize=13,
                 marker=marker_shape)  # Connect points with a line
        # print(i)
        # plt.plot(percentages, mae_values, '-o', label=feature, linewidth=2, markersize=8, )  # Connect points with a line

    for method, shape in source_free_methods_shapes.items():
        values = df[method].mean() * 100
        plt.scatter(0, values, label=methods_names[method], zorder=5, s=200, marker=shape)

    fontsize = 22
    plt.scatter(0, gnorm_gmm_mae, label='Our Method', zorder=5, s=300, marker="X", c="black")  # First point style

    plt.xlabel('Inclusion Ratio (%)', fontsize=fontsize + 8, fontweight='bold')
    plt.ylabel('MAE (%)', fontsize=fontsize + 8, fontweight='bold')
    plt.title(f'{dset_names[dset]}', fontsize=fontsize + 8, fontweight='bold')

    # plt.xticks(ticks=[0, 1, 5, 10, 20], fontsize=fontsize + 6, fontweight='bold')
    plt.xticks(ticks=[0, 100], fontsize=fontsize + 6, fontweight='bold')
    plt.yticks(fontsize=fontsize + 6, fontweight='bold')
    # position: top right, transparent background
    plt.legend(fontsize=fontsize, loc='upper right', framealpha=0.93 )
    plt.grid(True, alpha=0.2)
    # plt.savefig(f'./logs/plots/{dset}.png', dpi=300, bbox_inches='tight')
    return plt


def plot_boxplot_reach_gmm(smallest_indx, dname, dname_for_label, method_names_for_label,
                           fontsize=20):
    target_names = get_dset_corruption_names(dname)

    # Extracting names for the x-axis
    method_names = list(smallest_indx.keys())
    # Transposing the dictionary to group by index
    grouped_values = list(zip(*smallest_indx.values()))


    # Set the positions and width for the bars
    index = np.arange(len(grouped_values))

    opacity = 0.8
    bar_width = 0.09#/5
    bar_height = 0.17  # Adjust bar height as needed for clear visibility

    # Define patterns for different methods instead of colors
    patterns = ['/', '\\', '+', 'x', 'o']
    colors = ['#4F6D7A', '#C0D6DF', '#EAE8E9', '#B8C2BB', '#97AABD']

    # Create figure and axes for the patterns version, suitable for CVPR conference
    fig, ax = plt.subplots(figsize=(7, 5))
    plt.style.use('seaborn-v0_8-poster')
    # Adjusting the index for horizontal layout
    # Ensure you have defined group_names appropriately

    # Plotting each group with patterns horizontally
    for i, method in enumerate(method_names):
        #plt.barh(index + i * bar_height, [group[i]*100 for group in grouped_values], bar_height,
        #         alpha=opacity, label=method_names_for_label[method], hatch=patterns[i], edgecolor='black',
        #         color=colors[i])
        plt.bar(index + i*bar_width, [group[i]*100 for group in grouped_values], bar_width,
                alpha=opacity, label=method_names_for_label[method], hatch=patterns[i],
                edgecolor='black', )

    # Adding labels, title, and axes ticks
    # plt.ylabel(dname_for_label, fontsize=fontsize)  # Switched to ylabel for groups
    plt.xlim(left=0.0)
    plt.ylabel('Inclusion Ratio (%)', fontsize=fontsize-5,
               #labelpad=20, horizontalalignment='left', x=-0.15
               )  # Switched to xlabel for values
    # plt.title('Inclusion Ratio Required for Comparable Performance to Our Method', fontsize=fontsize)
    plt.title(dname_for_label, fontsize=fontsize)

    #plt.xticks(index + bar_height * 2,  target_names, fontsize=fontsize-5)  # Adjusted for horizontal bars
    #plt.yticks(fontsize=fontsize-5)
    plt.legend(fontsize=fontsize-7,# loc='lower right',
               framealpha=0.7
               )
    return fig, ax


def all_dsets_together_barplot(long_df, fontsize=20):
    # Set up the figure
    plt.figure(figsize=(16, 5))

    long_df = pd.DataFrame(long_df).stack().reset_index()

    long_df.columns = ['Dataset', 'Method', 'Percentage']

    # Unique datasets and methods for plotting
    datasets = long_df['Dataset'].unique()
    methods = long_df['Method'].unique()

    # Number of datasets and methods for calculating bar positions
    n_datasets = len(datasets)
    n_methods = len(methods)

    # Width of a bar group
    group_width = 0.35
    # Width of an individual bar within a group
    bar_width = group_width / n_methods
    # patterns = ['/', '\\', '+', 'x', 'o']
    colors =  ['powderblue', 'sandybrown', 'mediumseagreen', 'mediumpurple', 'burlywood']
    # colors = ['slateblue', 'sienna', 'olivedrab', 'indigo', 'brown']
    patterns =['/', '\\', '.', '-', '\\\\'] # Keeping the patterns for distinguishability
    # Define a wider space between the two groups by adjusting positions
    # Calculate new positions with an increased gap between the two groups
    gap_width = 0.2  # Increase the gap width between the groups
    new_positions = []
    temp=0
    for i in range(n_datasets):
        if i % 4==0 and i!=0:  # Digits group
            temp +=1
            print(i, temp)
        new_positions.append(i * (group_width + bar_width)+ gap_width * temp)
        # else:  # Wilds group, add the gap
        #     new_positions.append(i * (group_width + bar_width) + gap_width)

    # Create a bar plot for each method
    for index, method in enumerate(methods):
        # Calculate the position of each bar within the group
        adjusted_positions = [pos + (index - (n_methods - 1) / 2) * bar_width for pos in new_positions]

        # positions = [i + (index - (n_methods - 1) / 2) * bar_width for i in range(n_datasets)]

        # Filter the DataFrame for the current method
        method_data = long_df[long_df['Method'] == method]

        # Plot
        plt.bar(adjusted_positions, method_data['Percentage']*100, width=bar_width, label=method,
                #edgecolor='black',
                hatch=patterns[index],
                color=colors[index]
                )
    # Adjusting the x-axis limits to remove the space between the first bar and the y-axis
    plt.xlim(left=new_positions[0] - group_width*0.7)
    # Set the x-axis ticks to the dataset names, and rotate them for better readability
    print(datasets)
    # plt.xticks(range(n_datasets), datasets, fontsize=fontsize)
    plt.xticks(new_positions[0:n_datasets:1], datasets,  fontsize=fontsize-7)

    plt.yticks(fontsize=fontsize)
    # set y max
    # plt.ylim(0, 101)

    # Define additional group names and their positions
    # group_names = ['Digits', 'Wilds']

    group_names = ['PACS', 'VLCS', 'Office-Home', 'Terra Incognita']

    # Add additional group names with adjusted positions for clarity
    # group_positions_adjusted = [new_positions[1], (new_positions[3] +new_positions[4])/2 ]  # Manually adjust based on the new positions
    group_positions_adjusted = [( new_positions[1]+  new_positions[2])/2, (new_positions[5] + new_positions[6]) / 2,
                                (new_positions[9] + new_positions[10]) / 2, (new_positions[13] + new_positions[14]) / 2,
                                ]  # Manually adjust based on the new positions

    for pos, name in zip(group_positions_adjusted, group_names):
        plt.text(pos, -0.15, name, ha='center', transform=plt.gca().get_xaxis_transform(),
                 fontsize=fontsize)

    # Adding legend and labels
    # locate the legend outside of the plot
    # plt.legend(fontsize=fontsize-2, loc='upper center', bbox_to_anchor=(0.775, 0.7))
    plt.legend(fontsize=fontsize - 2, loc='upper center', bbox_to_anchor=(0.9, 0.7),
               framealpha=0.95
               )
    plt.ylabel('Inclusion Ratio (%)', fontsize=fontsize)
    # plt.title('Comparison of MAE Across Datasets for Different Methods')

    plt.tight_layout()
    plt.savefig('../logs/plots/boxplots/multi_source.png', dpi=300, bbox_inches='tight')




def main():
    dsets = ["pacs", "vlcs", "office_home", "terra_incognita"]
    methods_names = {
        "mae_ac": "AC", "mae_nuc_based": "Nuclear Norm",
        "mae_atc_entr": "ATC-Entropy", "mae_atc_prob": "ATC-Probability",
        "mae_cot_based": "COT", "mae_energy_based": "Energy",
        "mae_doc_based": "DOC"
    }

    dfs = []
    for dset in dsets:
        # read csv file
        df = pd.read_csv(f'..//logs/logs_v2/percentage_to_reach_gmm/percentage_to_reach_gnorm_{dset}.csv')
        # get the names of the datasets
        df.index = get_dset_corruption_names(dset)
        # replace colomn names with the method names
        for method in df.columns:
            df.rename(columns={method: methods_names[method]}, inplace=True)
        dfs.append(df)
    df = pd.concat(dfs)
    print(df)
    # transform into nested dictionary
    dict_df = df.to_dict()
    all_dsets_together_barplot(dict_df)
    print(df)

    print(df)
    # plt.show()


if __name__ =="__main__":
    main()