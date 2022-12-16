"""
This script runs a grid search on the experiments
"""
import json
import os

from analyses.truelearn_experiments.grid_search.generate_grid_search_config import EXP_OUTPUT_DIR_FIELD, \
    EXP_ALGORITHM_FIELD, EXP_DATASET_FILEPATH_FIELD, EXP_SKILL_REPR_FIELD, EXP_TAU_FACT_FIELD, EXP_BETA_FACT_FIELD, \
    EXP_THRESHOLD_FIELD, EXP_DEF_VAR_FACT_FIELD, EXP_ENGAGE_FUNC_FIELD, EXP_POSITIVE_ONLY_FIELD, EXP_DRAW_PROB_FIELD, \
    EXP_DRAW_FACTOR_FIELD, EXP_INT_DECAY_FACTOR_FIELD

EXP_ARGS = {EXP_DATASET_FILEPATH_FIELD, EXP_SKILL_REPR_FIELD, EXP_ALGORITHM_FIELD, EXP_TAU_FACT_FIELD,
            EXP_BETA_FACT_FIELD, EXP_THRESHOLD_FIELD, EXP_DEF_VAR_FACT_FIELD, EXP_ENGAGE_FUNC_FIELD,
            EXP_POSITIVE_ONLY_FIELD, EXP_OUTPUT_DIR_FIELD, EXP_DRAW_PROB_FIELD, EXP_DRAW_FACTOR_FIELD, EXP_INT_DECAY_FACTOR_FIELD}


def _get_exp_config_args(params):
    return {arg.replace("-", "_"): params.get(arg, arg.replace("_", "-")) for arg in EXP_ARGS}


def main(args):
    # read config file
    with open(args["configs_file"]) as infile:
        configs = [json.loads(l) for l in infile.readlines()]

    for idx, config in enumerate(configs):
        # create the results directory
        result_dir = args["results_dir"] + str(idx) + os.path.sep
        try:
            os.makedirs(result_dir)  # create the directory
        except:
            pass
        # if run experiments script is to be run:

        from analyses.truelearn_experiments.run_experiments import main as exp_main

        # create dir
        exp_result_dir = result_dir + os.path.sep
        try:
            os.makedirs(result_dir)  # create the directory
        except:
            pass

        config[EXP_OUTPUT_DIR_FIELD] = exp_result_dir

        # fingerprint configs here
        with open(exp_result_dir + "exp_config.json", "w") as outfile:
            json.dump(config, outfile)

        # run the grid search
        config_args = _get_exp_config_args(config)
        try:
            exp_main(config_args)
        except Exception as inst:
            print("Could not execute the run {} with config {} due to {}".format(idx, json.dumps(config), type(inst)))
            continue

        # run analyse results
        from analyses.truelearn_experiments.analyse_results import main as analyse_main

        for is_weigted_avg in [True, False]:
            temp_args = {
                "results_dir": exp_result_dir,
                "algorithm": config[EXP_ALGORITHM_FIELD],
                "weighted_avg": is_weigted_avg
            }

            analyse_main(temp_args)


if __name__ == '__main__':
    """this script takes in the wikified lectures file and the learner activity data from videolectures to build a .
    output of this script will be {slug, vid_id, part_id, start_time, stop_time, clean, text, wiki_concepts}
    eg: command to run this script:

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--configs-file', type=str, required=True,
                        help="the path to configs file where each line is a config dict")
    parser.add_argument('--results-dir', type=str, required=True,
                        help="directory where the results should be stored")

    args = vars(parser.parse_args())

    main(args)
