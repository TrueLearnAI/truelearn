import json
import os
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


def main(args):
    runs = [o for o in os.listdir(args["results_dir"])
            if os.path.isdir(os.path.join(args["results_dir"], o))]

    results = []

    for run in runs:
        result_path = os.path.join(args["results_dir"], run)

        try:
            basic_configs = _get_basic_configs(result_path)
        except Exception as inst:
            print("The run : {} was not created... skipping !!".format(run))
            continue

        basic_configs[RUN_ID_FIELD] = run

        algorithm = basic_configs[EXP_ALGORITHM_FIELD]

        try:

            (basic_configs["accuracy"], basic_configs["precision"],
             basic_configs["recall"], basic_configs["f1"]) = _get_accuracy_results(
                os.path.join(result_path, ACCURACY_FILEPATH), algorithm=algorithm)

            (basic_configs["accuracy_w"], basic_configs["precision_w"],
             basic_configs["recall_w"], basic_configs["f1_w"]) = _get_accuracy_results(
                os.path.join(result_path, ACCURACY_WEIGHTED_FILEPATH), algorithm=algorithm)

            results.append(basic_configs)

        except Exception as inst:
            print("The run : {} was not created... skipping !!".format(run))
            continue

    df = pd.DataFrame(results)
    df = df[
        ['run_id',
         EXP_SKILL_REPR_FIELD, EXP_TAU_FACT_FIELD, EXP_BETA_FACT_FIELD, EXP_DEF_VAR_FACT_FIELD, EXP_DRAW_PROB_FIELD,
         EXP_DRAW_FACTOR_FIELD, EXP_INT_DECAY_FACTOR_FIELD,
         'accuracy', 'precision', 'recall',
         'f1', 'accuracy_w', 'precision_w', 'recall_w', 'f1_w']]
    df.to_csv(os.path.join(args["results_dir"], "grid_search_results.csv"), index=False)


if __name__ == '__main__':
    """

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--results-dir', type=str, required=True,
                        help="directory where the results are stored")

    args = vars(parser.parse_args())

    main(args)
