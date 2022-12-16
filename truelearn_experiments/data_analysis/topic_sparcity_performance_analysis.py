import json
from os.path import join
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

truelearn_algos = ["truelearn_novel", "trueknowledge_sum_pos", "truelearn_background_pos"]
trueskill_algos = ["trueskill_single", "trueskill_multi"]


def main(args):
    # for normal records :/

    user_records = []
    for rec in os.walk(args["results_dir"]):
        root, _, files = rec
        if root.endswith("_test") and "userwise_metrics.csv" in files:
            for algo in truelearn_algos:
                if algo in root:
                    user_records.append(root)
                    break

    for root in user_records:
        filename = join(root, "userwise_metrics.csv")
        tmp_file = pd.read_csv(filename)[["f1", "num_events", "topic_sparsity_rate"]]
        tmp_file = tmp_file[tmp_file["f1"] > 0.]
        tmp_file["f1"] = tmp_file["f1"].apply(float)
        tmp_file.rename(
            columns={"f1": "F1 Score", "num_events": "Number of Events", "topic_sparsity_rate": "Topic Sparsity"},
            inplace=True)

        sns.set(style="ticks", palette="colorblind", context="paper")

        plt.figure()
        ax = sns.scatterplot(x="Topic Sparsity", y="Number of Events", hue="F1 Score",
                             size="F1 Score",
                             data=tmp_file,
                             legend="brief",
                             )
        plt.legend(fontsize='x-large')
        leg = ax.legend_
        for t in leg.texts:
            # truncate label text to 3 characters
            t.set_text(t.get_text()[:3])

        plt.savefig(join(root, "topic_sparsity.svg"), format="svg")

    # for trueskill
    user_records = []
    for rec in os.walk(args["results_dir"]):
        root, _, files = rec
        for algo in trueskill_algos:
            if algo in root:
                if "summary_results.csv" in files:
                    user_records.append(root)
                    break

    for root in user_records:
        filename = join(root, "summary_results.csv")
        tmp_file = pd.read_csv(filename)[["f1", "num_events", "topic_sparsity_rate"]]
        tmp_file = tmp_file[tmp_file["f1"] > 0.]
        tmp_file["f1"] = tmp_file["f1"].apply(float)
        tmp_file.rename(
            columns={"f1": "F1 Score", "num_events": "Number of Events", "topic_sparsity_rate": "Topic Sparsity"},
            inplace=True)

        sns.set(style="ticks", palette="colorblind", context="paper")

        plt.figure()
        ax = sns.scatterplot(x="Topic Sparsity", y="Number of Events", hue="F1 Score",
                             size="F1 Score",
                             data=tmp_file,
                             legend="brief",
                             )
        plt.legend(fontsize='x-large')
        leg = ax.legend_
        for t in leg.texts:
            # truncate label text to 3 characters
            t.set_text(t.get_text()[:3])

        plt.savefig(join(root, "topic_sparsity.svg"), format="svg")


if __name__ == '__main__':
    """this script takes in the wikified lectures file and the learner activity data from videolectures to build a .
    output of this script will be {slug, vid_id, part_id, start_time, stop_time, clean, text, wiki_concepts}
    eg: command to run this script:

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--results-dir', type=str, required=True,
                        help="the path to configs file where each line is a config dict")

    args = vars(parser.parse_args())

    main(args)
