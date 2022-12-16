import os
from os.path import join

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

_ALGO_ORDER = ['TrueLearn Novel',
               'Semantic TrueLearn Novel raw False',
               'Semantic TrueLearn Novel gauss False',
               'Semantic TrueLearn Novel pr False',
               'Semantic TrueLearn Novel raw True',
               'Semantic TrueLearn Novel gauss True',
               'Semantic TrueLearn Novel pr True']

_ALGO_MAPPING = {
    'TrueLearn Novel': 'TrueLearn',
    'Semantic TrueLearn Novel raw False': "ST \nSimple (P + I)",
    'Semantic TrueLearn Novel gauss False': "ST \nMultivariate (P + I)",
    'Semantic TrueLearn Novel pr False': "ST \nPageRank (P + I)",
    'Semantic TrueLearn Novel raw True': "ST \nSimple (P)",
    'Semantic TrueLearn Novel gauss True': "ST \nMultivariate (P)",
    'Semantic TrueLearn Novel pr True': "ST \nPageRank (P)",
}

_FEATURE_MAPPING = {'num_events': "Number of Events",
                    'num_topics_rate': "Topic Sparsity Rate",
                    'num_topics': "Number of Unique Topics",
                    'avg_sr_value': "Avg. SR Value",
                    'positive_rate': "Positive Label Rate",
                    'change_label_rate': "Label Change Rate",
                    'connectedness': "Average Connectedness",
                    'min_cut_set_size': "Min. Cut Set Size",
                    'num_conn_comps': "# of Connected Components",
                    'num_bridges': "# of Graph Bridges"}

_FEATURE_ORDER = ['num_events', 'num_topics', 'num_topics_rate',
                  'positive_rate',
                  # 'change_label_rate',
                  # 'avg_sr_value',
                  'connectedness', 'min_cut_set_size',
                  # 'num_conn_comps', 'num_bridges'
                  ]


def load_user_metrics(results_path, user_stats_path):
    res_df = pd.read_csv(join(results_path, "userwise_metrics.csv"))
    res_df["session"] = res_df["account_id"]
    res_df = res_df[["session", 'accuracy', 'precision', 'recall', 'f1', 'pr_score']]

    user_df = pd.read_json(user_stats_path)
    user_df = user_df[
        ['session', 'num_events', 'num_topics_rate', 'positive_rate', 'change_label_rate', 'num_topics', 'avg_sr_value',
         'connectedness', 'min_cut_set_size', 'num_conn_comps', 'num_bridges']]

    final_df = res_df.merge(user_df, on="session")
    final_df.sort_values(by="session", inplace=True)
    final_df.reset_index(drop=True, inplace=True)

    return final_df


def create_algorithm_name(algo, sr, pred):
    if algo == "semantic_truelearn":
        _algo = "Semantic TrueLearn Novel"
    else:
        return "TrueLearn Novel"

    if sr == "raw":
        _sr = "Uncorrelated"
    elif sr == "gauss":
        _sr = "Correlated"
    else:
        _sr = "PageRank-based"

    if pred:
        _pred = "Predict"
    else:
        _pred = "Update"

    return "{} {} {}".format(_algo, sr, pred)


def _get_significance_size(pvalue):
    if pvalue >= 0.05:
        return 0
    elif pvalue >= 0.01:
        return 0.0
    else:
        return 0.5


def get_data_structures(data_df, test="spearman"):
    correlations = {}
    pvalues = {}

    for idx, record in data_df.iterrows():
        if record["test"] == test:
            p_label = _get_significance_size(record["significance"])
            if p_label > 0.:
                pvalues[(record["algorithm"], record["feature"])] = p_label
                correlations[(record["algorithm"], record["feature"])] = record["correlation"]

    corr_matrix = np.zeros(shape=(len(_ALGO_ORDER), len(_FEATURE_ORDER)))
    pval_matrix = np.zeros(shape=(len(_ALGO_ORDER), len(_FEATURE_ORDER)))

    for algo_idx in range(len(_ALGO_ORDER)):
        for feat_idx in range(len(_FEATURE_ORDER)):
            corr = correlations.get((_ALGO_ORDER[algo_idx], _FEATURE_ORDER[feat_idx]))
            if corr is None:
                continue

            pvalue = pvalues[(_ALGO_ORDER[algo_idx], _FEATURE_ORDER[feat_idx])]

            corr_matrix[algo_idx, feat_idx] = corr
            pval_matrix[algo_idx, feat_idx] = pvalue

    return corr_matrix, pval_matrix


def create_meshplot(data_df):
    corr_matrix, pval_matrix = get_data_structures(data_df, test="spearman")

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.collections import PatchCollection

    a = pd.DataFrame(corr_matrix,
                     index=[_ALGO_MAPPING[alg] for alg in _ALGO_ORDER],
                     columns=[_FEATURE_MAPPING[ft] for ft in _FEATURE_ORDER])
    # ax = sns.heatmap(a, cmap="BuGn", annot=True)
    ax = sns.heatmap(a, cmap="Greens", annot=True)

    plt.xticks(rotation=30, ha='right')
    plt.show()
    print()

    algos = len(_ALGO_ORDER)
    features = len(_FEATURE_ORDER)

    ylabels = [_ALGO_MAPPING[alg] for alg in _ALGO_ORDER]  # y labels
    xlabels = [_FEATURE_MAPPING[ft] for ft in _FEATURE_ORDER]  # x labels

    x, y = np.meshgrid(np.arange(features), np.arange(algos))
    # pval_matrix = np.random.randint(0, 180, size=(algos, features))  # size of the circule
    # corr_matrix = np.random.rand(algos, features) - 0.5  # color of the grid

    fig, ax = plt.subplots()

    # R = pval_matrix / pval_matrix.max() / 2
    R = pval_matrix * 0.7
    circles = [plt.Rectangle((j, i), width=2 * r, height=2 * r) for r, j, i in zip(R.flat, x.flat, y.flat)]
    col = PatchCollection(circles, array=corr_matrix.flatten(), cmap="RdYlGn")
    ax.add_collection(col)

    ax.set(xticks=np.arange(features), yticks=np.arange(algos),
           xticklabels=xlabels, yticklabels=ylabels)
    ax.set_xticks(np.arange(features + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(algos + 1) - 0.5, minor=True)
    ax.grid(which='minor', linestyle='-', linewidth=1)

    plt.xticks(rotation=30, ha='right')
    fig.colorbar(col)
    plt.show()

    print()


def main(args):
    from analyses.truelearn_experiments.grid_search.eval.summarise_experiment_results import decode_algo_name
    # find what algos are there
    runs = [root for root, dirs, files in os.walk(args["results_dir_path"]) if "summary_metrics.csv" in files]

    all_records = []

    for run in runs:
        # decode algorithm config
        algorithm, agg_func, sr_func, pred_only, _ = decode_algo_name(run)

        user_metrics_df = load_user_metrics(run, args["user_stats_file_path"])

        for test in ["spearman", "pearson"]:
            for col in ['num_events', 'num_topics_rate',
                        # 'positive_rate',
                        'change_label_rate',
                        'num_topics',
                        'avg_sr_value', 'connectedness', 'min_cut_set_size',
                        # 'num_conn_comps', 'num_bridges'
                        ]:

                tmp_df = user_metrics_df.dropna(subset=[col])

                axis1 = tmp_df["recall"]
                axis2 = tmp_df[col]

                if test == "spearman":
                    result = spearmanr(axis1, axis2)
                    r, p = result.correlation, result.pvalue
                else:
                    r, p = pearsonr(axis1, axis2)

                if p < 0.05:
                    all_records.append({
                        "algorithm": create_algorithm_name(algorithm, sr_func, pred_only),
                        "feature": col,
                        "test": test,
                        "correlation": r,
                        "significance": p
                    }
                    )

    records_df = pd.DataFrame(all_records)

    # Create_visualization
    create_meshplot(records_df)


if __name__ == '__main__':
    """Through this script, we want to check the relationship between topic coverage. We investigate the relationship 
    between engagement and normalised topic coverage for top 5 most prominent subjects in the 
    eg: command to run this script:

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--results-dir-path', type=str, required=True,
                        help="where the result files are")
    parser.add_argument('--user-stats-file-path', type=str, required=True,
                        help="where the result files are")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Output file path where the results will be saved.")

    args = vars(parser.parse_args())

    main(args)
