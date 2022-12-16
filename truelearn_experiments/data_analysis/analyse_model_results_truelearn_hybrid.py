import os
from os.path import join
import ujson as json

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from analyses.truelearn_experiments.data_analysis.generate_userwise_analyses import get_user_stats, run_paired_t_test

import seaborn as sns
from matplotlib import pyplot as plt

from lib.spark_context import get_spark_context

sns.set_palette("colorblind")


def get_records(records, window):
    keys = set()

    stats = []
    for _, record in records.iterrows():
        predicted = [float(i >= .5) for i in record["predicted"]]
        actual = record["actual"]
        sess = record["session"]

        breaker = max(len(actual) + 1, window + 1)

        for end in range(window, breaker):
            start = end - window
            tmp_pred = predicted[start:end]
            tmp_act = actual[start:end]

            tmp_acc = accuracy_score(tmp_act, tmp_pred, normalize=True)
            tmp_pre = precision_score(tmp_act, tmp_pred)
            tmp_rec = recall_score(tmp_act, tmp_pred)
            tmp_f1 = f1_score(tmp_act, tmp_pred)

            keys.add(end)

            stats.append({
                "session": sess,
                "events": end,
                "accuracy": tmp_acc,
                "precision": tmp_pre,
                "recall": tmp_rec,
                "f1": tmp_f1,
            })

    stats_df = pd.DataFrame(stats)

    tmp_df = stats_df[['events', 'accuracy', 'precision', 'recall', 'f1']].groupby("events").mean().reset_index()

    return tmp_df


def get_weights(line, n):
    weights = line["weights"]
    if n >= len(weights):
        line["w_k"] = None
        line["w_i"] = None
    else:
        current_weight = weights[n]
        if type(current_weight) == list:
            line["w_k"] = current_weight[0]
            line["w_i"] = current_weight[1]
            # line["w_k"] = np.abs(current_weight[0])
            # line["w_i"] = np.abs(current_weight[1])
        else:
            line["w_k"] = current_weight

    return line


def extract_user_summary(data):
    num_events = max(data["num_events"])

    items = []

    for n in range(101):
        tmp_df = data[data["num_events"] > n]
        tmp_df = tmp_df.apply(lambda l: get_weights(l, n), axis=1).dropna(subset=["w_k", "w_i"])
        # diff = tmp_df["w_k"] - tmp_df["w_i"]
        # n_k = sum(diff > 0) / len(diff)

        avg_wk = np.mean(tmp_df["w_k"])
        avg_wi = np.mean(tmp_df["w_i"])

        items.extend([
            {"num_events": n, "weight": "w_k", "value": avg_wk},
            {"num_events": n, "weight": "w_i", "value": avg_wi}
        ])

    item_df = pd.DataFrame(items)

    item_df.dropna()

    sns.set_palette("colorblind")
    plt.clf()
    ax = sns.lineplot(x="num_events", y="value", hue="weight", data=item_df)
    plt.show()


def get_weights_only(line):
    weights = line["weights"][-1]
    sum_weights = sum(weights)
    line["w_k"], line["w_i"] = weights[0] / 1, weights[1] / 1

    return line


def calculate_agreement(l):
    k_probs = np.array(l["k_probs"]) >= .5
    i_probs = np.array(l["i_probs"]) >= .5

    agreement = accuracy_score(k_probs, i_probs, normalize=True)

    l["agreement"] = agreement

    return l


def generate_topic_sparsity_plot(data, norm=False):
    new_data = data.apply(get_weights_only, axis=1)

    new_data["k_positive_rate"] = new_data["k_probs"].apply(lambda l: np.mean(np.array(l) >= .5))
    new_data["i_positive_rate"] = new_data["i_probs"].apply(lambda l: np.mean(np.array(l) >= .5))

    new_data = new_data.apply(calculate_agreement, axis=1)

    topic_rate_label = "num_user_topics"

    new_data["topic_rate_difference"] = new_data['i_num_user_topics'] - new_data['num_user_topics']

    r, p = pearsonr(new_data["w_k"], new_data[topic_rate_label])
    print("r:{}, p:{}".format(r, p))
    r, p = pearsonr(new_data["w_i"], new_data[topic_rate_label])
    print("r:{}, p:{}".format(r, p))

    new_data[topic_rate_label] = (np.round(new_data[topic_rate_label] * .010) / .010).astype("int")

    if norm:
        new_data["normaliser"] = new_data["w_k"] + new_data["w_i"]
    else:
        new_data["normaliser"] = 1

    tmp_df_k = new_data[[topic_rate_label]]
    tmp_df_k["value"] = new_data["w_k"] / new_data["normaliser"]
    tmp_df_k["Weight"] = "W_k"

    tmp_df_i = new_data[[topic_rate_label]]
    tmp_df_i["value"] = new_data["w_i"] / new_data["normaliser"]
    tmp_df_i["Weight"] = "W_i"

    tmp_df = tmp_df_k.append(tmp_df_i)
    # ax = sns.boxplot(x="i_num_topics_rate_cat", y="value", hue="Weight", data=tmp_df)
    ax = sns.lineplot(x=topic_rate_label, y="value", hue="Weight", data=tmp_df)
    plt.savefig(join(args["output_dir"], "Truelearn_hybrid_greedy_weight_vs_unique_topics.svg"), format="svg")

    # for sparcity in ["num_topics_rate", "i_num_topics_rate"]:
    #     plt.clf()
    #     tmp_data = new_data[["w_k", "w_i", sparcity]]#
    #     ax = sns.scatterplot(x="w_k", y="w_i", hue=sparcity, data=tmp_data)
    #     plt.show()

    print()


def generate_event_records(line, tot_events):
    num_events = min(line["num_events"] - 1, 50)
    for idx in range(num_events):
        id = int(line["session"])
        event_num = idx + 1
        actual = int(line["actual"][idx])
        k_pred = int(line["k_probs"][idx] >= .5)
        i_pred = int(line["i_probs"][idx] >= .5)
        pred = int(line["predicted"][idx] >= .5)

        weights = line["weights"][idx]
        w_k = float(weights[0])
        w_i = float(weights[1])

        sample_weight = float(line["num_events"] / tot_events)

        yield (event_num, (id, actual, k_pred, i_pred, pred, w_k, w_i, sample_weight))


def compute_metrics(records):
    event_num, records = records[0], sorted(list(records[1]), key=lambda k: k[0])  #

    _, actual, k_pred, i_pred, pred, w_k, w_i, sample_weight = zip(*records)

    acc_temp_dict = {
        "Hybrid Accuracy": float(accuracy_score(actual, pred, normalize=True, sample_weight=sample_weight)),
        "Knowledge Accuracy": float(accuracy_score(actual, k_pred, normalize=True, sample_weight=sample_weight)),
        "Interest Accuracy": float(accuracy_score(actual, i_pred, normalize=True, sample_weight=sample_weight)),
        # "Gap" : float(np.abs(accuracy_score(actual, k_pred, normalize=True, sample_weight=sample_weight) -
        #                      accuracy_score(actual, i_pred, normalize=True, sample_weight=sample_weight))),
        # "Avg. W_k": float(np.mean(w_k)),
        # "Avg. W_i": np.mean(w_i),
    }

    prec_temp_dict = {
        "Hybrid Precision": float(precision_score(actual, pred, sample_weight=sample_weight)),
        "Knowledge Precision": float(precision_score(actual, k_pred, sample_weight=sample_weight)),
        "Interest Precision": float(precision_score(actual, i_pred, sample_weight=sample_weight)),
        # "Gap": float(np.abs(precision_score(actual, k_pred, sample_weight=sample_weight) -
        #                     precision_score(actual, i_pred, sample_weight=sample_weight))),
        # "Avg. W_k": float(np.mean(w_k)),
        # "Avg. W_i": np.mean(w_i),
    }

    rec_temp_dict = {
        "Hybrid Recall": float(recall_score(actual, pred, sample_weight=sample_weight)),
        "Knowledge Recall": float(recall_score(actual, k_pred, sample_weight=sample_weight)),
        "Interest Recall": float(recall_score(actual, i_pred, sample_weight=sample_weight)),
        # "Gap": float(np.abs(recall_score(actual, k_pred, sample_weight=sample_weight) -
        #                     recall_score(actual, i_pred, sample_weight=sample_weight))),
        # "Avg. W_k": float(np.mean(w_k)),
        # "Avg. W_i": np.mean(w_i),
    }

    f1_temp_dict = {
        "Hybrid F1": float(f1_score(actual, pred, sample_weight=sample_weight)),
        "Knowledge F1": float(f1_score(actual, k_pred, sample_weight=sample_weight)),
        "Interest F1": float(f1_score(actual, i_pred, sample_weight=sample_weight)),
        # "Gap": float(np.abs(f1_score(actual, k_pred, sample_weight=sample_weight) -
        #                     f1_score(actual, i_pred, sample_weight=sample_weight))),
        # "Avg. W_k": float(np.mean(w_k)),
        # "Avg. W_i": np.mean(w_i),
    }

    for k, v in acc_temp_dict.items():
        yield {"Event Number": int(event_num), "Metric": k, "Value": v, "Type": "Accuracy"}

    for k, v in prec_temp_dict.items():
        yield {"Event Number": int(event_num), "Metric": k, "Value": v, "Type": "Precision"}

    for k, v in rec_temp_dict.items():
        yield {"Event Number": int(event_num), "Metric": k, "Value": v, "Type": "Recall"}

    for k, v in f1_temp_dict.items():
        yield {"Event Number": int(event_num), "Metric": k, "Value": v, "Type": "F1"}


def main(args):
    if args["report"] == "ttest":
        runs = [root for root, dirs, files in os.walk(args["results_dir_path"]) if "model_results.json" in files]

        mapping = {}

        for cat in ["knowledge", "both"]:
            path = [i for i in runs if cat in i][0]
            user_stats = get_user_stats(path)
            mapping[cat] = user_stats

        # for metric in ["precision", "recall", "f1"]:
        for metric in ["precision", "recall", "f1"]:
            t, p = run_paired_t_test(mapping["knowledge"][[metric]], mapping["both"][[metric]])
            print(p)
    elif args["report"] == "weights":
        runs = [root for root, dirs, files in os.walk(args["results_dir_path"]) if "summary_metrics.csv" in files]

        for algo in [
            # "meta_logistic",
            # "meta_perceptron",
            # "meta_truelearn",
            # "meta_trueLearn_greedy"
            "meta_trueLearn_greedy_with_bias"
        ]:
            run = [run for run in runs if algo in run][0]
            data = pd.read_json(join(run, "model_results.json"))
            extract_user_summary(data)

        print()

    elif args["report"] == "weights-topics":
        runs = [root for root, dirs, files in os.walk(args["results_dir_path"]) if "summary_metrics.csv" in files]
        data_df = pd.DataFrame()
        for algo in [
            # "meta_logistic",
            # "meta_perceptron",
            # "meta_truelearn",
            # "meta_trueLearn_greedy"
            "meta_trueLearn_greedy_with_bias"
        ]:
            # for cat in ["knowledge", "interest", "both"]:
            run_path = [join(run, "model_results.json") for run in runs if algo in run]
            if len(run_path) == 0:
                continue

            tmp_df = pd.read_json(run_path[0])
            print(np.mean(tmp_df["predict_positive_rate"] * tmp_df["num_events"] / (tmp_df["num_events"] - 1)))

            tmp_df["video_accuracy"] = tmp_df["vid_pos_changes"] / tmp_df["vid_changes"]
            np.mean(tmp_df["video_accuracy"].dropna())  # calulating video change accuracy

            generate_topic_sparsity_plot(tmp_df)
            print()

            # tmp_df["event_cat"] = np.floor(tmp_df["num_events"] / 10).astype("int")
            tmp_df["model"] = algo

            data_df = data_df.append(tmp_df, ignore_index=True)


    elif args["report"] == "events-accuracy":
        spark = get_spark_context()

        filepath = [root for root, dirs, files in os.walk(args["results_dir_path"]) if
                    "summary_metrics.csv" in files and "meta_trueLearn_greedy_with_bias" in root][0]

        with open(join(filepath, "model_results.json")) as infile:
            data = json.load(infile)

        total_events = sum(l["num_events"] for l in data)

        data_rdd = spark.sparkContext.parallelize(data)
        # test = data_rdd.first()
        #
        # tt = list(generate_event_records(test))
        stats_data = data_rdd.flatMap(lambda l: generate_event_records(l, total_events)).groupByKey().flatMap(
            compute_metrics).collect()

        stats_df = pd.DataFrame(stats_data)

        for metric in ["Accuracy", "Precision", "Recall", "F1"]:
            tmp_df = stats_df[stats_df["Type"] == metric]
            plt.clf()
            ax = sns.lineplot(x="Event Number", y="Value", hue="Metric", data=tmp_df)
            plt.savefig(join(args["output_dir"], "Truelearn_hybrid_greedy_{}.svg".format(metric)), format="svg")

        print()


    else:
        runs = [root for root, dirs, files in os.walk(args["results_dir_path"]) if "model_results.json" in files]
        data_df = pd.DataFrame()
        for cat in ["knowledge", "interest"]:
            # for cat in ["knowledge", "interest", "both"]:
            run_path = [join(run, "model_results.json") for run in runs if cat in run]
            if len(run_path) == 0:
                continue

            tmp_df = pd.read_json(run_path[0])
            print(np.mean(tmp_df["predict_positive_rate"] * tmp_df["num_events"] / (tmp_df["num_events"] - 1)))

            tmp_df["video_accuracy"] = tmp_df["vid_pos_changes"] / tmp_df["vid_changes"]
            np.mean(tmp_df["video_accuracy"].dropna())  # calulating video change accuracy

            tmp_df = tmp_df[tmp_df["num_events"] <= 100]
            tmp_df = get_records(tmp_df, 10)

            # tmp_df["event_cat"] = np.floor(tmp_df["num_events"] / 10).astype("int")
            tmp_df["model"] = cat

            data_df = data_df.append(tmp_df, ignore_index=True)

        for metric in ["precision", "recall", "f1"]:
            ax = sns.lineplot(x="events", y=metric, hue="model", data=data_df)

            plt.show()


if __name__ == '__main__':
    """Through this script, we want to check the relationship between topic coverage. We investigate the relationship 
    between engagement and normalised topic coverage for top 5 most prominent subjects in the 
    eg: command to run this script:

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--results-dir-path', type=str, required=True,
                        help="where the result files are")
    parser.add_argument('--report', default='short', const='all', nargs='?',
                        choices=['ttest', "weights", "knowint", "weights-topics", "events-accuracy"])
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Output file path where the results will be saved.")

    args = vars(parser.parse_args())

    main(args)
