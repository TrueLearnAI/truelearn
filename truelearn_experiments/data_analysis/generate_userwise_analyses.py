import os
from os.path import join

import pandas as pd
import numpy as np

from scipy.stats import ttest_rel

np.random.seed(42)

from analyses.truelearn_experiments.grid_search.eval.summarise_experiment_results import decode_algo_name


def get_user_stats(path):
    try:
        user_results = pd.read_csv(join(path, "userwise_metrics.csv"))
    except FileNotFoundError:
        user_results = pd.read_csv(join(path, "summary_results.csv"))

    user_results = user_results[user_results["num_events"] >= 5]

    user_results.sort_values(by="account_id", inplace=True)
    user_results.reset_index(drop=True, inplace=True)

    return user_results


def run_paired_t_test(baseline, algo):

    t, p = ttest_rel(algo, baseline, alternative="greater") # algo > baseline hypothesis test
    return t[0], p[0]


def get_summary_results(result_df):
    stat_dict = {}
    stat_dict["accuracy_w"] = np.average(result_df["accuracy"], weights=result_df["num_events"])
    stat_dict["precision_w"] = np.average(result_df["precision"], weights=result_df["num_events"])
    stat_dict["recall_w"] = np.average(result_df["recall"], weights=result_df["num_events"])
    stat_dict["f1_w"] = np.average(result_df["f1"], weights=result_df["num_events"])

    return [stat_dict]


def main(args):
    # find what algos are there
    runs = [root for root, dirs, files in os.walk(args["results_dir_path"])
            if "summary_metrics.csv" in files or "summary_results.csv" in files]

    algorithms = []
    baseline_path = None

    for run in runs:
        # decode algorithm config
        algorithm, agg_func, sr_func, pred_only, _ = decode_algo_name(run)

        if algorithm == "baseline_truelearn":
            baseline_path = run
        else:
            algorithms.append({
                "name": algorithm,
                "filepath": run,
                "sr_func": sr_func,
                "pred_only": pred_only
            })

    # load baseline user_stats
    baseline_user_stats = get_user_stats(baseline_path)
    w_performance = pd.DataFrame(get_summary_results(baseline_user_stats))
    w_performance.to_csv(join(baseline_path, "weighted_performance.csv"), index=False)

    for algorithm in algorithms:
        result_path = algorithm["filepath"]
        algo_user_stats = get_user_stats(result_path)

        w_performance = pd.DataFrame(get_summary_results(algo_user_stats))
        w_performance.to_csv(join(result_path, "weighted_performance.csv"), index=False)

        # run paired t-test
        t_acc, p_acc = run_paired_t_test(baseline_user_stats[["accuracy"]], algo_user_stats[["accuracy"]])
        t_prec, p_prec = run_paired_t_test(baseline_user_stats[["precision"]], algo_user_stats[["precision"]])
        t_rec, p_rec = run_paired_t_test(baseline_user_stats[["recall"]], algo_user_stats[["recall"]])
        t_f1, p_f1 = run_paired_t_test(baseline_user_stats[["f1"]], algo_user_stats[["f1"]])

        algorithm["acc_t"] = t_acc
        algorithm["acc_p"] = p_acc
        algorithm["prec_t"] = t_prec
        algorithm["prec_p"] = p_prec
        algorithm["rec_t"] = t_rec
        algorithm["rec_p"] = p_rec
        algorithm["f1_t"] = t_f1
        algorithm["f1_p"] = p_f1

    t_stat_df = pd.DataFrame(algorithms)
    t_stat_df = t_stat_df[
        ['name', 'sr_func', 'pred_only', 'acc_t', 'acc_p', 'prec_t', 'prec_p', 'rec_t', 'rec_p', 'f1_t', 'f1_p']]

    t_stat_df.sort_values(by=["name", "pred_only", "sr_func"], inplace=True)

    t_stat_df.to_csv(os.path.join(args["results_dir_path"], "grid_search_t_results.csv"), index=False)


if __name__ == '__main__':
    """this script takes in the wikified lectures file and the learner activity data from videolectures to build a .
    output of this script will be {slug, vid_id, part_id, start_time, stop_time, clean, text, wiki_concepts}
    eg: command to run this script:

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--results-dir-path', type=str, required=True,
                        help="where the result files are")

    # parser.add_argument('--skill-repr', default='cosine', const='all', nargs='?',
    #                     choices=['cosine', 'binary', "norm"],
    #                     help="How the skills should be represented in the models")
    # parser.add_argument('--output-dir', type=str, required=True,
    #                     help="Output directory path where the results will be saved.")
    # parser.add_argument('--engage-func', default='all', const='all', nargs='?',
    #                     choices=['all', 'sum', "quality"],
    #                     help="What engagement eval function to be used")
    # parser.add_argument('--threshold', type=float, default=1.,
    #                     help="Probability threshold for classifying true")
    # parser.add_argument('--def-var-factor', type=float, default=.5,
    #                     help="Probability of knowing this topics")
    # parser.add_argument('--tau-factor', type=float, default=.0,
    #                     help="Probability of watching even when cant learn")
    # parser.add_argument('--interest-decay-factor', type=float, default=.0,
    #                     help="Probability of watching even when cant learn")
    # parser.add_argument('--beta-factor', type=float, default=.1,
    #                     help="Probability skipping even when can learn")
    # parser.add_argument('--draw-probability', type=str, default="static",
    #                     help="Probability of drawing the match")
    # parser.add_argument('--draw-factor', type=float, default=.1,
    #                     help="factor of draw probability to be used")
    # parser.add_argument("--num-topics", type=int, default=10,
    #                     help="The number of top ranked topics that have to be considered.")
    # parser.add_argument('--positive-only', action='store_true', help="learns from negative examples too")
    # parser.add_argument('--has-part-id', action='store_true', help="defines if the dataset has part ids")

    args = vars(parser.parse_args())

    _ = main(args)
