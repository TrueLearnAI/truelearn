import json
from os.path import join

import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt
sns.color_palette("tab20")

def get_metric_data(path, label):
    data = pd.read_csv(path)
    data["algorithm"] = label
    return data


def main(args):
    # read directories
    data = pd.DataFrame()
    data = data.append(get_metric_data(join(args["results_json_dir"], "ink", "_test", "metrics.csv"), "truelearn_ink"), ignore_index=True)
    data = data.append(get_metric_data(join(args["results_json_dir"], "qink_switched", "_test", "metrics.csv"), "truelearn_pink_switch"), ignore_index=True)
    data = data.append(get_metric_data(join(args["results_json_dir"], "qink_weighted", "_test", "metrics.csv"), "truelearn_ink_weight"), ignore_index=True)

    # plot them together
    for m in ["Accuracy", "Precision", "Recall", "F1-Score"]:
    # for m in ["Accuracy", "Precision"]:
        temp_df = data[data["Metric"] == m]
        plt.clf()
        ax = sns.barplot(x="x", y="Value", hue="algorithm", data=temp_df, palette="tab20")
        ax.set(ylim=(0.3, 1.))
        ax.set_title("Metric: {}".format(m))
        plt.savefig(join(args["results_json_dir"], "{}.svg".format(m)))

    print()

if __name__ == '__main__':
    """this script takes in the wikified lectures file and the learner activity data from videolectures to build a .
    output of this script will be {slug, vid_id, part_id, start_time, stop_time, clean, text, wiki_concepts}
    eg: command to run this script:

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--results-json-dir', type=str, required=True,
                        help="the path the json file dir that contains results")
    parser.add_argument('--n-jobs', type=str, default="*",
                        help="number of parallel jobs")
    # parser.add_argument('--test', default='vid-freq', const='all', nargs='?',
    #                     choices=['vid-freq', 'skill-freq', 'n-events'],
    #                     help="Type of fucntion used to combine knowledge and interest")

    args = vars(parser.parse_args())

    main(args)