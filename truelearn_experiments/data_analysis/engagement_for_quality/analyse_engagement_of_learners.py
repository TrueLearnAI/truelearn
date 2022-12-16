import json
from os.path import join

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from lib.spark_context import get_spark_context

import seaborn as sns
from matplotlib import pyplot as plt


def _get_person_video_stats(record):
    vid_change_np = np.array(record["video_change"])
    act_np = np.array(record["actual"])
    pred_np = np.array(record["predicted"]) >= 0.5

    # get the indexes of all the video changes
    vid_change_idxs = np.where(vid_change_np == 1)[0]

    for idx, vid_change_idx in enumerate(vid_change_idxs):
        # rounded_idx = int((np.ceil(idx/5))*5)
        rounded_idx = idx
        if rounded_idx > 25:
            break
        yield (rounded_idx, [(int(act_np[vid_change_idx]), int(pred_np[vid_change_idx]))])


def compute_metrics(act, pred):
    stats = {}
    stats["Accuracy"] = accuracy_score(act, pred)
    stats["Precision"] = precision_score(act, pred)
    stats["Recall"] = recall_score(act, pred)
    stats["F1-Score"] = f1_score(act, pred)

    return stats


def get_performance_stats(records):
    vid_number, stats = records
    act, pred = map(list, zip(*stats))

    stats = compute_metrics(act, pred)

    for m, val in stats.items():
        yield {"n": int(vid_number), "Metric": m, "Value": float(val)}


def get_video_frequency_statistics(data):
    list(_get_person_video_stats(data.collect()[2]))
    vid_stats = (data.
                 flatMap(_get_person_video_stats).
                 reduceByKey(lambda a, b: a + b).
                 flatMap(get_performance_stats)).collect()

    return vid_stats


def get_update_stats(updates, act, pred, cat="k"):
    for idx in range(len(updates)):
        a = act[idx]
        p = pred[idx]
        u = updates[idx]

        yield ((cat, "min", int(min(u))), [(a, p)])
        yield ((cat, "max", int(max(u))), [(a, p)])
        yield ((cat, "avg", int(round(np.mean(u)))), [(a, p)])
        yield ((cat, "sum", int(sum(u))), [(a, p)])


def get_person_skill_stats(record):
    act = [int(i) for i in record["actual"]]
    pred = [int(i) for i in np.array(record["predicted"]) >= .5]
    # get knowledge updates
    k_updates = record["num_updates"]
    k_lines = list(get_update_stats(k_updates, act, pred, "k"))

    # get interest updates
    i_updates = record["i_num_updates"]
    i_lines = list(get_update_stats(i_updates, act, pred, "i"))

    return k_lines + i_lines


def get_person_n_stats(record):
    act = [int(i) for i in record["actual"]]
    pred = [int(i) for i in np.array(record["predicted"]) >= .5]

    for idx in range(len(act)):
        yield (idx, [(act[idx], pred[idx])])


def get_update_performance_stats(record):
    (cat, summ, num), stats = record
    act, pred = map(list, zip(*stats))

    stats = compute_metrics(act, pred)

    for m, val in stats.items():
        yield {"cat": cat, "summary": summ, "x": num, "Metric": m, "Value": float(val)}

def get_n_event_performance_stats(record):
    n, stats = record
    act, pred = map(list, zip(*stats))

    stats = compute_metrics(act, pred)

    for m, val in stats.items():
        yield {"x": n + 1, "Metric": m, "Value": float(val)}

def get_skill_freq_statistics(data):
    stat_terms = (data.
                  flatMap(get_person_skill_stats).
                  reduceByKey(lambda a, b: a + b).
                  flatMap(get_update_performance_stats)).collect()

    return stat_terms


def get_event_n_statistics(data):
    stat_terms = (data.
                  flatMap(get_person_n_stats).
                  reduceByKey(lambda a, b: a + b).
                  flatMap(get_n_event_performance_stats)).collect()

    return stat_terms


def main(args):
    spark = get_spark_context(master="local[{}]".format(args["n_jobs"]))
    spark.sparkContext.setLogLevel("ERROR")

    with open(join(args["results_json_dir"], "model_results.json")) as infile:
        profiles = json.load(infile)

    # Load the dataset
    user_data = (spark.sparkContext.parallelize(profiles).repartition(30))

    # do the analyses
    if args["test"] == "vid-freq":
        # Vid freq vs. Acc, prec, rec, F1
        vid_freq_stats = get_video_frequency_statistics(user_data)

        stat_df = pd.DataFrame(vid_freq_stats)

        for m in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            temp_df = stat_df[stat_df["Metric"] == m]
            temp_df = temp_df[temp_df["n"] <= 10]

            plt.clf()
            ax = sns.barplot(x="n", y="Value", data=temp_df, color='steelblue')
            plt.show()
    elif args["test"] == "skill-freq":
        skill_freq_stats = get_skill_freq_statistics(user_data)
        stat_df = pd.DataFrame(skill_freq_stats)

        for cat in ["k", "i"]:
            # for summ in ["min", "max", "avg", "sum"]:
            for summ in ["sum"]:
                for m in ["Accuracy", "Precision", "Recall", "F1-Score"]:
                    temp_df = stat_df[stat_df["Metric"] == m]
                    temp_df = temp_df[temp_df["cat"] == cat]
                    temp_df = temp_df[temp_df["summary"] == summ]
                    temp_df = temp_df[temp_df["x"] <= 20]

                    plt.clf()
                    ax = sns.barplot(x="x", y="Value", data=temp_df)
                    ax.set_title("Signal: {}, Summary: {}, Metric: {}".format(cat, summ, m))
                    plt.show()

    elif args["test"] == "n-events":
        event_n_stats = get_event_n_statistics(user_data)
        stat_df = pd.DataFrame(event_n_stats)

        save_df = pd.DataFrame()

        for m in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            temp_df = stat_df[stat_df["Metric"] == m]
            temp_df = temp_df[temp_df["x"] <= 10]
            save_df = save_df.append(temp_df, ignore_index = True)
            plt.clf()
            ax = sns.barplot(x="x", y="Value", data=temp_df, color='steelblue')
            ax.set(ylim=(0.3, 1.))
            ax.set_title("Metric: {}".format(m))
            plt.show()
            # plt.savefig(join(args["results_json_dir"], "{}.svg".format(m)))

        save_df.to_csv(join(args["results_json_dir"], "metrics.csv"), index=False)
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
    parser.add_argument('--test', default='vid-freq', const='all', nargs='?',
                        choices=['vid-freq', 'skill-freq', 'n-events'],
                        help="Type of fucntion used to combine knowledge and interest")

    args = vars(parser.parse_args())

    main(args)
