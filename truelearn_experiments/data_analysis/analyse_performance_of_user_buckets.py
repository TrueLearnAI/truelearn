from os.path import join

import numpy as np
import pandas as pd


def load_user_data(path, algo):
    data = pd.read_csv(path)
    data["event_bucket"] = data["num_events"].apply(lambda x: np.ceil(x / 50) * 50)

    tmp_vals = []
    for buck in data["event_bucket"].unique():
        vals = data[data["event_bucket"] == buck]["recall"]
        rec_mean = np.mean(vals)
        tmp_vals.append({
            "Number of Events: Groups of Learners": int(buck),
            "Mean Recall Score": rec_mean,
            "Algorithm": algo
        })

    return tmp_vals


def main(args):
    baseline_data = load_user_data(
        join(args["results_dir_path"], "truelearn_novel_neg", "_test", "userwise_metrics.csv"),
        "TrueLearn Novel")

    st_simp = load_user_data(
        join(args["results_dir_path"], "semantic_truelearn_novel_neg_or", "_test", "userwise_metrics.csv"),
        "ST Simple (P + I)")
    st_mult = load_user_data(
        join(args["results_dir_path"], "semantic_truelearn_novel_neg_or_gauss", "_test", "userwise_metrics.csv"),
        "ST Multivariate (P + I)")
    st_pr = load_user_data(
        join(args["results_dir_path"], "semantic_truelearn_novel_neg_or_pr", "_test", "userwise_metrics.csv"),
        "ST PageRank (P + I)")

    st_simp_p = load_user_data(
        join(args["results_dir_path"], "semantic_truelearn_novel_neg_or_pred", "_test", "userwise_metrics.csv"),
        "ST Simple (P)")
    st_mult_p = load_user_data(
        join(args["results_dir_path"], "semantic_truelearn_novel_neg_or_pred_gauss", "_test", "userwise_metrics.csv"),
        "ST Multivariate (P)")
    st_pr_p = load_user_data(
        join(args["results_dir_path"], "semantic_truelearn_novel_neg_or_pred_pr", "_test", "userwise_metrics.csv"),
        "ST PageRank (P)")

    full_df = pd.DataFrame(baseline_data + st_simp + st_mult + st_pr)
    # + st_simp_p + st_mult_p + st_pr_p)

    import seaborn as sns
    from matplotlib import pyplot as plt

    ax = sns.lineplot(x="Number of Events: Groups of Learners", y="Mean Recall Score", hue="Algorithm", data=full_df)
    print()


if __name__ == '__main__':
    """Through this script, we want to check the relationship between topic coverage. We investigate the relationship 
    between engagement and normalised topic coverage for top 5 most prominent subjects in the 
    eg: command to run this script:

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--results-dir-path', type=str, required=True,
                        help="where the result files are")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Output file path where the results will be saved.")

    args = vars(parser.parse_args())

    main(args)
