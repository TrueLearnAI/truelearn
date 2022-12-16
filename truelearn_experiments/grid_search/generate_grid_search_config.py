import json
from copy import copy

import numpy as np

EXP_THRESHOLD_FIELD = "threshold"
EXP_DEF_VAR_FACT_FIELD = "def_var_factor"
EXP_DRAW_PROB_FIELD = "draw_probability"
EXP_DRAW_FACTOR_FIELD = "draw_factor"
EXP_ENGAGE_FUNC_FIELD = "engage_func"
EXP_DATASET_FILEPATH_FIELD = "dataset_filepath"
EXP_SKILL_REPR_FIELD = "skill_repr"
EXP_ALGORITHM_FIELD = "algorithm"
EXP_OUTPUT_DIR_FIELD = "output_dir"
EXP_TAU_FACT_FIELD = "tau_factor"
EXP_BETA_FACT_FIELD = "beta_factor"
EXP_POSITIVE_ONLY_FIELD = "positive_only"
EXP_INT_DECAY_FACTOR_FIELD = "interest_decay_factor"

N_ITER_FIELD = "n_iter"

np.random.seed(seed=42)

_DEF_TRUE_LEARN_BEARNAULLI_DEFAULTS = {
    EXP_ALGORITHM_FIELD: "truelearn_background",
    EXP_ENGAGE_FUNC_FIELD: "all",
}

_DEF_TRUE_LEARN_GAUSSIAN_DEFAULTS = {
    EXP_ALGORITHM_FIELD: "trueknowledge_sum",
    EXP_ENGAGE_FUNC_FIELD: "all",
}

_DEF_TRUE_LEARN_GAUSSIAN_ALL_DEFAULTS = {
    EXP_ALGORITHM_FIELD: "trueknowledge_all",
    EXP_ENGAGE_FUNC_FIELD: "quality",
}

_DEF_TRUE_LEARN_NOVEL_DEFAULTS = {
    EXP_ALGORITHM_FIELD: "truelearn_novel",
    EXP_ENGAGE_FUNC_FIELD: "all",
}

_DEF_TRUE_LEARN_NOVEL_INTEREST_DEFAULTS = {
    EXP_ALGORITHM_FIELD: "truelearn_novel_interest",
    EXP_ENGAGE_FUNC_FIELD: "all",
}


def get_exp_params(params):
    # only one filepath is allowed
    filepath = params["exp_" + EXP_DATASET_FILEPATH_FIELD]
    skill_repr = params["exp_" + EXP_SKILL_REPR_FIELD]
    pos_only = params["exp_" + EXP_POSITIVE_ONLY_FIELD]

    # tau_facts = np.arange(0, 1.01, 0.2)  # p_guess_grid search_20_users
    # beta_facts = np.arange(0, 1.01, 0.2)  # p_fail_grid search_1_20 users

    tau_facts = np.arange(0, .41, 0.2)  # p_guess_grid search_full_data
    beta_facts = np.arange(0.5, 1.01, 0.2)  # p_fail_grid search_full_data
    draw_probs = ["static"]
    thresholds = [0.5]
    draw_facts = [.0]
    interest_decay_factor = [.0]

    if args["exp_" + EXP_ALGORITHM_FIELD] == "truelearn_background":
        def_fields = _DEF_TRUE_LEARN_BEARNAULLI_DEFAULTS
        def_vars = [0.5]
        tau_facts = tau_facts * 0.3
        beta_facts = beta_facts * 0.3
    else:
        if args["exp_" + EXP_ALGORITHM_FIELD] == "trueknowledge_sum":
            def_fields = _DEF_TRUE_LEARN_GAUSSIAN_DEFAULTS
        elif args["exp_" + EXP_ALGORITHM_FIELD] == "truelearn_novel":
            def_fields = _DEF_TRUE_LEARN_NOVEL_DEFAULTS
            draw_probs = ["individual"]
            draw_facts = [.01, .05, .1]
            tau_facts = [.005, .01, .05]
        elif args["exp_" + EXP_ALGORITHM_FIELD] == "truelearn_novel_interest":
            def_fields = _DEF_TRUE_LEARN_NOVEL_INTEREST_DEFAULTS
            draw_probs = ["individual"]
            draw_facts = [.05]
            tau_facts = [.0]
            interest_decay_factor = [0.0001, 0.0005, .001, .005, .01, 0.05]
        else:
            def_fields = _DEF_TRUE_LEARN_GAUSSIAN_ALL_DEFAULTS

        # beta_facts = [0.25, 0.5, 1., 2., 5., 10.] # grid_search_20_users
        beta_facts = [.5]  # grid_search_all_users

        # def_vars = [0.25, 0.5, 1., 2., 5., 10.] # grid_search_20_users
        def_vars = [1000.]  # grid_search_all_users

    records = []

    for def_var in def_vars:
        for tau_factor in tau_facts:
            for beta_factor in beta_facts:
                for thresh in thresholds:
                    for draw_prob in draw_probs:
                        for draw_fact in draw_facts:
                            for int_fact in interest_decay_factor:
                                fields = copy(def_fields)
                                fields[EXP_DEF_VAR_FACT_FIELD] = def_var
                                fields[EXP_TAU_FACT_FIELD] = tau_factor
                                fields[EXP_BETA_FACT_FIELD] = beta_factor
                                fields[EXP_DATASET_FILEPATH_FIELD] = filepath
                                fields[EXP_SKILL_REPR_FIELD] = skill_repr
                                fields[EXP_POSITIVE_ONLY_FIELD] = pos_only
                                fields[EXP_THRESHOLD_FIELD] = thresh
                                fields[EXP_DRAW_PROB_FIELD] = draw_prob
                                fields[EXP_DRAW_FACTOR_FIELD] = draw_fact
                                fields[EXP_INT_DECAY_FACTOR_FIELD] = int_fact

                                records.append(fields)

    return records


def main(args):
    output_filepath = args.pop("configs_file")

    # get lists of params for exp
    configs = get_exp_params(args)

    with open(output_filepath, "w") as out:
        config_str = "\n".join([json.dumps(l) for l in configs])
        out.write(config_str)


if __name__ == '__main__':
    """generates a config file where every line is a combination of parameters  

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--exp-dataset-filepath', type=str, required=True,
                        help="filepath where the data is")
    parser.add_argument('--exp-skill-repr', type=str, required=True,
                        help="How the skills should be represented in the models")
    parser.add_argument('--exp-algorithm', type=str, required=True,
                        help="algorithm name")
    parser.add_argument('--exp-positive-only', action='store_true',
                        help="if learning should be done from positive engagement aswell")
    parser.add_argument('--configs-file', type=str, required=True,
                        help="the path to save the configs file where each line is a config dict")

    args = vars(parser.parse_args())

    main(args)
