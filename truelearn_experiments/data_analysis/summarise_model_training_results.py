import json
from os.path import join
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

truelearn_algos = ["persistent", "majority", "truelearn_novel", "trueknowledge_sum_pos", "trueknowledge_sum_neg", "truelearn_background_pos", "truelearn_background_neg"]
trueskill_algos = ["trueskill_single", "trueskill_multi"]


def main(args):
    # for normal records :/

    final_results = []
    user_records = []
    for rec in os.walk(args["results_dir"]):
        root, _, files = rec
        if root.endswith("_test") and "summary_metrics.csv" in files:
            for algo in truelearn_algos:
                if algo in root:
                    user_records.append((root, algo))
                    break

    for root, algo in user_records:
        filename = join(root, "summary_metrics.csv")
        tmp_file = pd.read_csv(filename)[["accuracy_w", "precision_w", "recall_w", "f1_w"]]
        record = dict(tmp_file.iloc[0])
        record["algorithm"] = algo
        final_results.append(record)

    # for trueskill
    user_records = []
    for rec in os.walk(args["results_dir"]):
        root, _, files = rec
        for algo in trueskill_algos:
            if algo in root:
                if "summary_accuracy_weighted.csv" in files:
                    user_records.append((root, algo))
                    break

    for root, algo in user_records:
        filename = join(root, "summary_accuracy_weighted.csv")
        tmp_file = pd.read_csv(filename)[["accuracy_w", "precision_w", "recall_w", "f1_w"]]
        record = dict(tmp_file.iloc[0])
        record["algorithm"] = algo
        final_results.append(record)

    final_df = pd.DataFrame(final_results)[["algorithm", "accuracy_w", "precision_w", "recall_w", "f1_w"]]

    print()


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
