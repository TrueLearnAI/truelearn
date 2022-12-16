import json
from os.path import join, split
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

_ALGO_SET = {
    "cosine", "jaccard_c", "knowledge_tracing", "truelearn_novel"
}


def get_tag(path, tag_set):
    root, tail = split(path)
    if not tail in tag_set and root != "":
        return get_tag(root, tag_set)
    else:
        return tail


def get_algorithm(path):
    algo = get_tag(path, _ALGO_SET)

    assert algo in _ALGO_SET

    return algo


def get_topics(path):
    tag_set = {
        "1_topics", "2_topics", "3_topics", "5_topics"
    }

    num_topics = get_tag(path, tag_set)

    assert num_topics in tag_set

    num, _ = num_topics.split("_")

    return int(num)


def extract_details(path):
    algorithm = get_algorithm(path)
    num_topics = get_topics(path)

    return algorithm, num_topics


def extract_results(path):
    # read file
    results = pd.read_csv(join(path, "summary_metrics.csv"))
    # extract required items
    prec = results["precision_w"][0] * 100.
    rec = results["recall_w"][0] * 100.
    f1 = results["f1_w"][0] * 100.

    return prec, rec, f1


def create_plot(records, out_dir):
    import seaborn as sns
    from matplotlib import pyplot as plt
    sns.color_palette("colorblind")

    plt.clf()
    plt.rcParams["figure.figsize"] = [20, 5]
    f, axes = plt.subplots(1, 3)

    for idx, metric in enumerate(["Precision", "Recall", "F1-Score"]):
        tmp_df = records[records["Evaluation Metric"] == metric]
        sns.lineplot(data=tmp_df,
                     x="No. of Topics", y="Value", hue="Algorithm", style="Algorithm",
                     markers=True, dashes=False,
                     ax=axes[idx])

    plt.savefig(join(out_dir, "results_plot.svg"), format="svg")


def main(args):
    runs = [root for root, dirs, files in os.walk(args["results_dir"]) if "summary_metrics.csv" in files]
    records = []
    for run in runs:
        if "jaccard_u" in run:
            continue

        algorithm, num_topics = extract_details(run)

        prec, rec, f1 = extract_results(run)

        records.extend([
            {"Algorithm": algorithm, "No. of Topics": num_topics, "Evaluation Metric": "Precision", "Value": prec},
            {"Algorithm": algorithm, "No. of Topics": num_topics, "Evaluation Metric": "Recall", "Value": rec},
            {"Algorithm": algorithm, "No. of Topics": num_topics, "Evaluation Metric": "F1-Score", "Value": f1}
        ])

    records_df = pd.DataFrame(records)

    create_plot(records_df, args["output_dir"])


if __name__ == '__main__':
    """this script takes in the wikified lectures file and the learner activity data from videolectures to build a .
    output of this script will be {slug, vid_id, part_id, start_time, stop_time, clean, text, wiki_concepts}
    eg: command to run this script:

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--results-dir', type=str, required=True,
                        help="the path to configs file where each line is a config dict")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="the path to configs file where each line is a config dict")
    args = vars(parser.parse_args())

    main(args)
