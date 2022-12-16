import os
from collections import defaultdict
from os.path import join
import ujson as json
import numpy as np
import pandas as pd

from analyses.truelearn_experiments.data_analysis.analyse_user_stats_performance_relationship import \
    create_algorithm_name
from analyses.truelearn_experiments.grid_search.eval.summarise_experiment_results import decode_algo_name

_ALGO_MAPPING = {
    'TrueLearn Novel': 'TrueLearn',
    'Semantic TrueLearn Novel raw False': "ST \nSimple (P + I)",
    'Semantic TrueLearn Novel gauss False': "ST \nMultivariate (P + I)",
    'Semantic TrueLearn Novel pr False': "ST \nPageRank (P + I)",
    'Semantic TrueLearn Novel raw True': "ST \nSimple (P)",
    'Semantic TrueLearn Novel gauss True': "ST \nMultivariate (P)",
    'Semantic TrueLearn Novel pr True': "ST \nPageRank (P)",
}


def calculate_average(times):
    vals = times
    # med = np.median(times)
    # threshold = 3 * med
    # vals = [i for i in times if i <= threshold]
    return np.mean(vals)


def label_event_category(num_events):
    if num_events < 20:
        return 20
    elif num_events < 40:
        return 40
    elif num_events < 60:
        return 60
    elif num_events < 80:
        return 80
    elif num_events < 100:
        return 100
    else:
        return 120


def label_algorithm(algo, sr, pred):
    return _ALGO_MAPPING[create_algorithm_name(algo, sr, pred)]


def load_user_data(path):
    with open(join(path, "model_results.json")) as infile:
        user_profiles = json.load(infile)

    relevant_records = []
    for user in user_profiles:
        times = user.pop("total_duration")
        if times is not None:
            avg_time = calculate_average(times)
            time_per_event = float(avg_time / user["num_events"])

            relevant_records.append({
                "num_events": user["num_events"],
                "Time Per Event (ms)": time_per_event
            })

    return relevant_records


def main(args):
    # find what algos are there
    runs = [root for root, dirs, files in os.walk(args["results_dir_path"]) if "userwise_metrics.csv" in files]

    all_records = pd.DataFrame()

    num_users = None

    for run in runs:
        # decode algorithm config
        algorithm, agg_func, sr_func, pred_only, _ = decode_algo_name(run)
        algo_name = label_algorithm(algorithm, sr_func, pred_only)

        # load file
        user_data = load_user_data(run)

        user_data_df = pd.DataFrame(user_data)

        user_data_df["Number of Events"] = user_data_df["num_events"].apply(label_event_category)
        user_data_df["Algorithm"] = algo_name

        num_users = user_data_df[["Number of Events", "num_events"]].groupby('Number of Events').count()

        all_records = all_records.append(user_data_df, ignore_index=True)
        print(len(all_records))

    import seaborn as sns
    from matplotlib import pyplot as plt
    ax1 = sns.lineplot(
        data=all_records, x='Number of Events', y="Time Per Event (ms)", hue='Algorithm', err_style="bars", ci=68
    )

    # num_users["Number of Learners"] = num_users["num_events"]

    # ax2= ax1.twinx()
    #
    # sns.lineplot(data=num_users, x='Number of Events', y="Number of Learners", style=True, dashes=[(2,2)], color="grey",
    #              marker="o", legend=False
    #              )

    plt.show()
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
