import json
import os
from os.path import split

import pandas as pd

RUN_ID_FIELD = "run_id"
ACCURACY_FILEPATH = "summary_accuracy.csv"
ACCURACY_WEIGHTED_FILEPATH = "summary_accuracy_weighted.csv"
RECALL = "{}_recall"
PRECISION = "{}_precision"
ACCURACY = "{}_accuracy"
F1 = "{}_f1"

from analyses.truelearn_experiments.grid_search.generate_grid_search_config import EXP_SKILL_REPR_FIELD, \
    EXP_ALGORITHM_FIELD, EXP_TAU_FACT_FIELD, EXP_BETA_FACT_FIELD, EXP_DEF_VAR_FACT_FIELD, EXP_DRAW_PROB_FIELD, \
    EXP_DRAW_FACTOR_FIELD, EXP_INT_DECAY_FACTOR_FIELD

RELEVANT_RESULT_FIELDS = {EXP_SKILL_REPR_FIELD, EXP_ALGORITHM_FIELD, EXP_SKILL_REPR_FIELD, EXP_ALGORITHM_FIELD,
                          EXP_TAU_FACT_FIELD, EXP_BETA_FACT_FIELD, EXP_DEF_VAR_FACT_FIELD, EXP_DRAW_PROB_FIELD,
                          EXP_DRAW_FACTOR_FIELD, EXP_INT_DECAY_FACTOR_FIELD}


def _get_basic_configs(path):
    with open(os.path.join(path, "exp_config.json")) as infile:
        temp_dict = json.load(infile)

    return {k: v for k, v in temp_dict.items() if k in RELEVANT_RESULT_FIELDS}


def _get_accuracy_results(filepath, algorithm):
    df = pd.read_csv(filepath)
    return (df[ACCURACY.format(algorithm)][0], df[PRECISION.format(algorithm)][0], df[RECALL.format(algorithm)][0],
            df[F1.format(algorithm)][0])


def decode_algo_name(filepath):
    leaf = split(filepath)[-1]
    if leaf == "_test":
        folder_path = split(filepath)[0]
    else:
        folder_path = filepath

    algo_name = split(folder_path)[-1]

    algorithm = "baseline_truelearn"
    agg_func = "raw"
    pred_only = False

    algo_name_parts = algo_name.split("_")

    if "semantic" in algo_name_parts:
        algorithm = "semantic_truelearn"
    if "or" in algo_name_parts:
        agg_func = "or"
    if "pred" in algo_name_parts:
        pred_only = True

    if "pr" in algo_name_parts:
        sr_func = "pr"
    elif "gauss" in algo_name_parts:
        sr_func = "gauss"
    else:
        sr_func = "raw"

    return algorithm, agg_func, sr_func, pred_only, algo_name


def main(args):
    # runs = [o for o in os.listdir(args["results_dir"])
    #         if os.path.isdir(os.path.join(args["results_dir"], o))]

    runs = [root for root, dirs, files in os.walk(args["results_dir"]) if "summary_metrics.csv" in files]

    summary = []

    for run in runs:
        # decode algorithm config
        algorithm, agg_func, sr_func, pred_only, algo_name = decode_algo_name(run)

        # read_config
        record = pd.read_csv(os.path.join(run, "best_hyperparameters.csv"))
        record = record.to_dict('records')[0]

        # read results
        results = pd.read_csv(os.path.join(run, "summary_metrics.csv"))
        results = results.to_dict('records')[0]

        record = {**record, **results}

        record["algorithm"] = algorithm
        record["agg_func"] = agg_func
        record["sr_func"] = sr_func
        record["pred_only"] = pred_only
        record["dir_name"] = algo_name

        summary.append(record)

    summary = pd.DataFrame(summary)

    if args["hybrid"]:
        summary = summary[
            ['algorithm', 'agg_func', 'sr_func', 'pred_only', 'def_var_factor', 'i_def_var_factor',
             'know_prob', 'interest_decay_type', 'interest_decay_factor',
             'draw_probability', 'draw_factor',
             'accuracy', 'precision', 'recall', 'f1', "roc_score", "pr_score",
             'accuracy_w', 'precision_w', 'recall_w', 'f1_w', "roc_score_w", "pr_score_w"]]
    else:
        summary = summary[
            ['algorithm', 'agg_func', 'sr_func', 'pred_only', 'def_var_factor', 'draw_probability', 'draw_factor',
             'accuracy', 'precision', 'recall', 'f1', "roc_score", "pr_score",
             'accuracy_w', 'precision_w', 'recall_w', 'f1_w', "roc_score_w", "pr_score_w"]]

    summary.sort_values(by=["algorithm", "pred_only", "sr_func"], inplace=True)

    summary.to_csv(os.path.join(args["results_dir"], "model_results_summary.csv"), index=False)


if __name__ == '__main__':
    """

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--results-dir', type=str, required=True,
                        help="directory where the results are stored")
    parser.add_argument('--hybrid', action='store_true', help="if the models are hybrid")

    args = vars(parser.parse_args())

    main(args)
