from os import makedirs
from os.path import join

import numpy as np
import pandas as pd

_final_def_var = {
    # algorithm, sr_func, pred_only, full_dataset
    ###### small dataset
    ("semantic_truelearn_novel", "gauss", False, False): [100],
    ("semantic_truelearn_novel", "pr", False, False): [1000],
    ("semantic_truelearn_novel", "raw", False, False): [1000],

    ("semantic_truelearn_novel", "gauss", True, False): [100],
    ("semantic_truelearn_novel", "pr", True, False): [5000],
    ("semantic_truelearn_novel", "raw", True, False): [100],

    ######  full dataset
    ("semantic_truelearn_novel", "gauss", False, True): [150000],
    ("semantic_truelearn_novel", "pr", False, True): [150000],
    ("semantic_truelearn_novel", "raw", False, True): [150000],

    ("semantic_truelearn_novel", "gauss", True, True): [10000],
    ("semantic_truelearn_novel", "pr", True, True): [50000],
    ("semantic_truelearn_novel", "raw", True, True): [50000],

    # algorithm , full dataset
    ("truelearn_novel", "raw", False, False): [250],
    ("truelearn_novel", "raw", False, True): [1000]
}


# _threshold_vals = [.05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95]


def generate_hyperparameters(algorithm, top_n=10, pos_only=False, sem_path=None, pred_only=False, agg_func="raw",
                             dilute_var=False, top_k_sr_topics=1, sr_func="raw", is_full_dataset=False,
                             is_final=False, is_timing=False, topics=False, prob_combine_type="weight",
                             source=None, quality_path=None, quality_type="k", q_random=False):
    ranges = {}
    skill_repr = None

    if algorithm in {"cbf", "jaccard"}:
        skill_repr = "cosine"
        ranges = {
            "def_var_factor": [0.],
            "i_def_var_factor": [0],
            "var_constant": [0],
            "beta_factor": [0],
            "tau_factor": [0],
            "interest_decay_type": ["short"],
            "interest_decay_factor": [0],
            "draw_probability": ["static"],
            "draw_factor": [1.],
            "dilution_factor": [0],
            "know_prob": [0],
            "num_signals": [0],
            "freq_agg": ["k"]
        }

    elif algorithm in {'ccf', 'user_tfidf'}:
        skill_repr = "binary"
        ranges = {
            "def_var_factor": [0.],
            "i_def_var_factor": [0],
            "var_constant": [0],
            "beta_factor": [0],
            "tau_factor": [0],
            "interest_decay_type": ["short"],
            "interest_decay_factor": [0],
            "draw_probability": ["static"],
            "draw_factor": [1.],
            "dilution_factor": [0],
            "know_prob": [0],
            "num_signals": [0],
            "freq_agg": ["k"]
        }

    elif algorithm in {"user_interest_cos", "user_interest_bin"}:
        if "cos" in algorithm:
            skill_repr = "cosine"
        else:
            skill_repr = "binary"

        algorithm = "user_interest"

        ranges = {
            "def_var_factor": [0.],
            "i_def_var_factor": [0],
            "var_constant": [0],
            "beta_factor": [0],
            "tau_factor": [0],
            "interest_decay_type": ["short"],
            "interest_decay_factor": [0],
            "draw_probability": ["static"],
            "draw_factor": [1.],
            "dilution_factor": [0],
            "know_prob": [0],
            "num_signals": [0],
            "freq_agg": ["k"]
        }



    elif algorithm == "knowledge_tracing":
        skill_repr = "binary"
        ranges = {
            "def_var_factor": [.000001],  # pinit
            "var_constant": [0.],  # for full dataset
            "beta_factor": np.arange(.0, 1.01, 0.2) * .3,  # pfail
            "tau_factor": np.arange(.0, 1.01, 0.2) * .3,  # pguess
            "draw_probability": ["static"],
            "draw_factor": [1.],
            "interest_decay_type": ["short"],
            "interest_decay_factor": [.0],
            "dilution_factor": [0],
            "know_prob": [0],
            "num_signals": [0],
            "freq_agg": ["k"]
        }

    elif algorithm == "knowledge_tracing_interest":
        skill_repr = "binary"
        ranges = {
            "def_var_factor": [.000001],  # pinit
            "var_constant": [0.],  # for full dataset
            "beta_factor": np.arange(.0, 1.01, 0.2) * .3,  # pfail
            "tau_factor": np.arange(.0, 1.01, 0.2) * .3,  # pguess
            "draw_probability": ["static"],
            "draw_factor": [1.],
            "interest_decay_type": ["short"],
            "interest_decay_factor": [0.0],
            "dilution_factor": [0.0],
            "know_prob": [0],
            "num_signals": [0],
            "freq_agg": ["k"]
        }


    elif algorithm == "truelearn_fixed":
        skill_repr = "cosine"

        if is_full_dataset:
            def_var_factor = [750, 1500, 2500, 5000, 50000, 100000]  # for full dataset

            ranges = {
                "def_var_factor": def_var_factor,
                "var_constant": [0],  # for full dataset
                "beta_factor": [.5],
                "tau_factor": [0.],
                "interest_decay_type": ["short"],
                "interest_decay_factor": [.0],
                "draw_probability": ["static"],
                "draw_factor": [1.],
                "num_signals": [0],
                "freq_agg": ["k"]
            }
        else:
            def_var_factor = [750, 1500, 2500, 5000]

            ranges = {
                "def_var_factor": def_var_factor,
                "var_constant": [0],  # for full dataset
                "beta_factor": [.5],
                "tau_factor": [0.],
                "interest_decay_type": ["short"],
                "interest_decay_factor": [.0],
                "draw_probability": ["static"],
                "draw_factor": [1.],
                "num_signals": [0],
                "freq_agg": ["k"]
            }

    elif algorithm == "semantic_truelearn_fixed":
        skill_repr = "cosine"

        if is_full_dataset:
            def_var_factor = [250000, 500000, 1000000, 2000000, 3000000, 4000000]  # for full dataset

            ranges = {
                "def_var_factor": def_var_factor,
                "var_constant": [0],  # for full dataset
                "beta_factor": [.5],
                "tau_factor": [0.],
                "interest_decay_type": ["short"],
                "interest_decay_factor": [.0],
                "draw_probability": ["static"],
                "draw_factor": [1.],
                "dilution_factor": [.0],
                "num_signals": [0],
                "freq_agg": ["k"]
            }
        else:
            def_var_factor = [100, 250, 750, 1500, 2500, 5000, 10000, 25000]

            ranges = {
                "def_var_factor": def_var_factor,
                "var_constant": [0],  # for full dataset
                "beta_factor": [.5],
                "tau_factor": [0.],
                "interest_decay_type": ["short"],
                "interest_decay_factor": [.0],
                "draw_probability": ["static"],
                "draw_factor": [1.],
                "dilution_factor": [.0],
                "num_signals": [0],
                "freq_agg": ["k"]
            }

    elif algorithm == "truelearn_novel":
        skill_repr = "cosine"

        if is_full_dataset:
            # def_var_factor = [100, 500, 1000, 2500, 4000, 5000, 7500, 10000]  # all topics take 2
            def_var_factor = [1000, 2500, 3000, 4000, 5000]  # all topics take 2

            ranges = {
                "def_var_factor": def_var_factor,
                "var_constant": [0],  # for full dataset
                "beta_factor": [.5],
                "tau_factor": [0.],
                "interest_decay_type": ["short"],
                "interest_decay_factor": [.0],
                "draw_probability": ["static"],
                # "draw_probability": ["individual"],
                # "draw_factor": [.25, 0.5, 0.75, 1.]
                "draw_factor": [1.],
                "num_signals": [0],
                "freq_agg": ["k"]

            }
        else:
            # def_var_factor = [75, 100, 250, 500, 1000]  # for 2-5 topics
            def_var_factor = [500, 750, 1000, 2000, 5000]  # for 1 and 10 topics

            ranges = {
                "def_var_factor": def_var_factor,
                "var_constant": [0],  # for full dataset
                "beta_factor": [.5],
                "tau_factor": [0.],
                "interest_decay_type": ["short"],
                "interest_decay_factor": [.0],
                "draw_probability": ["static"],
                # "draw_probability": ["individual"],
                # "draw_factor": [.25, 0.5, 0.75, 1.]
                "draw_factor": [1.],
                "num_signals": [0],
                "freq_agg": ["k"]

            }

    elif algorithm == "truelearn_novelq_pop_pred":
        skill_repr = "cosine"

        if is_full_dataset:
            # def_var_factor = [100, 500, 1000, 2500, 4000, 5000, 7500, 10000]  # all topics take 2
            def_var_factor = [4000]  # all topics take 2

            ranges = {
                "def_var_factor": def_var_factor,
                "var_constant": [0],  # for full dataset
                "beta_factor": [.5],
                "tau_factor": [0.],
                "interest_decay_type": ["short"],
                "interest_decay_factor": [.0],
                "draw_probability": ["static"],
                # "draw_probability": ["individual"],
                # "draw_factor": [.25, 0.5, 0.75, 1.]
                "draw_factor": [1.],
                "num_signals": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "freq_agg": ["k"]

            }
        else:
            # def_var_factor = [75, 100, 250, 500, 1000]  # for 2-5 topics
            def_var_factor = [500, 750, 1000, 2000, 5000]  # for 1 and 10 topics

            ranges = {
                "def_var_factor": def_var_factor,
                "var_constant": [0],  # for full dataset
                "beta_factor": [.5],
                "tau_factor": [0.],
                "interest_decay_type": ["short"],
                "interest_decay_factor": [.0],
                "draw_probability": ["static"],
                # "draw_probability": ["individual"],
                # "draw_factor": [.25, 0.5, 0.75, 1.]
                "draw_factor": [1.],
                "num_signals": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "freq_agg": ["k"]

            }

    elif algorithm == "semantic_truelearn_novel":
        skill_repr = "cosine"

        if is_full_dataset:
            # def_var_factor = [10000, 25000, 50000, 60000]  # for full dataset pred only
            # def_var_factor = [125000, 150000, 200000, 300000]  # for full dataset pred and learn

            # def_var_factor = [10, 25, 50, 100]  # for full dataset
            def_var_factor = [5000, 10000, 250000, 50000, 100000, 150000, 200000]

            ranges = {
                "def_var_factor": def_var_factor,  # for full dataset
                "var_constant": [0],
                "beta_factor": [.5],
                "tau_factor": [0.],
                "interest_decay_type": ["short"],
                "interest_decay_factor": [.0],
                "draw_probability": ["static"],

                # "draw_probability": ["individual"],
                # "draw_factor": [0.1, .25], # 5 topics
                # "draw_factor": [.05, .1, .25, .5, .75], # 1 topic
                "draw_factor": [1.],
                "dilution_factor": [0.0],
                "num_signals": [0],
                "freq_agg": ["k"]
            }

        else:
            def_var_factor = [75, 100, 500, 1000, 2500, 5000, 7500]  # for top 20

            ranges = {
                "def_var_factor": def_var_factor,  # for full dataset
                "var_constant": [0],
                "beta_factor": [.5],
                "tau_factor": [0.],
                "interest_decay_type": ["short"],
                "interest_decay_factor": [.0],
                "draw_probability": ["static"],

                # "draw_probability": ["individual"],
                # "draw_factor": [0.1, .25], # 5 topics
                # "draw_factor": [.05, .1, .25, .5, .75], # 1 topic
                "draw_factor": [1.],
                "dilution_factor": [0.0],
                "num_signals": [0],
                "freq_agg": ["k"]
            }

    elif algorithm == 'persistent' or algorithm == 'majority':
        skill_repr = "binary"
        ranges = {
            "def_var_factor": [0],
            "var_constant": [0],  # for full dataset
            "beta_factor": [0],
            "tau_factor": [0],
            "interest_decay_type": ["short"],
            "interest_decay_factor": [0],
            "draw_probability": ["individual"],
            "draw_factor": [0.],
            "num_signals": [0],
            "freq_agg": ["k"]
        }

    elif algorithm == "truelearn_interest":
        skill_repr = "binary"

        if is_full_dataset:
            # def_var_factor = [100000, 250000, 500000]  # topics >= 5
            # def_var_factor = [100, 250, 500, 1000, 2500, 5000, 10000, 100000]  # 250, 500, 1000
            def_var_factor = [100, 250, 500]

            ranges = {
                "def_var_factor": def_var_factor,
                "var_constant": [0],  # for full dataset
                "beta_factor": [.5],
                "tau_factor": [0.],
                # "interest_decay_type": ["short", "long"],
                # "interest_decay_factor": [0.0, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2],
                "interest_decay_type": ["short"],
                "interest_decay_factor": [0.0],
                # "draw_probability": ["static", "individual"],
                "draw_probability": ["static"],
                # "draw_factor": [0.1, .25, 0.5, 0.75, 1.]
                "draw_factor": [1.],
                "num_signals": [0],
                "freq_agg": ["k"]
            }
        else:
            def_var_factor = [1, 3, 5, 7, 10, 50, 75, 100, 200, 350, 500]  # for top 20

            ranges = {
                "def_var_factor": def_var_factor,
                "var_constant": [0],  # for full dataset
                "beta_factor": [.5],
                "tau_factor": [0.],
                "interest_decay_type": ["short"],
                "interest_decay_factor": [0.0],
                "draw_probability": ["static"],
                # "draw_probability": ["individual"],
                # "draw_factor": [.25, 0.5, 0.75, 1.]
                "draw_factor": [1.],
                "num_signals": [0],
                "freq_agg": ["k"]
            }

    elif algorithm == "semantic_truelearn_interest":
        skill_repr = "binary"

        if is_full_dataset:
            # def_var_factor = [10, 25, 50, 100]  # for full dataset
            def_var_factor = [2500, 5000, 10000]

            ranges = {
                "def_var_factor": def_var_factor,  # for full dataset
                "var_constant": [0],
                "beta_factor": [.5],
                "tau_factor": [0.],
                "interest_decay_type": ["short", "long"],
                "interest_decay_factor": [0.005, 0.01, 0.05, 0.1, 0.15, 0.2],
                # "interest_decay_factor": [.0, 0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 0.9],
                "draw_probability": ["static"],
                "draw_factor": [1.],
                "dilution_factor": [0.0],
                "num_signals": [0],
                "freq_agg": ["k"]
            }
        else:
            def_var_factor = [10, 25, 50, 75, 100]  # for top 20

            ranges = {
                "def_var_factor": def_var_factor,  # for full dataset
                "var_constant": [0],
                "beta_factor": [.5],
                "tau_factor": [0.],
                "interest_decay_type": ["short", "long"],
                "interest_decay_factor": [0.005, 0.01, 0.05, 0.1, 0.15, 0.2],
                # "interest_decay_factor": [.0, 0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 0.9],
                "draw_probability": ["static"],
                "draw_factor": [1.],
                "dilution_factor": [0.0],
                "num_signals": [0],
                "freq_agg": ["k"]
            }
    elif algorithm == "truelearn_hybrid":
        skill_repr = "cosine"

        if is_full_dataset:
            # def_var_factor = [250, 500, 750]  # 1 topic
            # def_var_factor = [5000, 10000, 25000, 50000]  # 3 topics
            def_var_factor = [5000]

            # i_def_var_factor = [250, 500, 1000] # 1 topic
            # i_def_var_factor = [7, 10, 25, 50, 75, 100]  # 5 topic
            i_def_var_factor = [250]

            ranges = {
                "def_var_factor": def_var_factor,
                "i_def_var_factor": i_def_var_factor,
                "var_constant": [0],
                "beta_factor": [.5],
                "tau_factor": [.0],
                "interest_decay_type": ["short"],
                "interest_decay_factor": [.0],
                "draw_probability": ["static"],
                "draw_factor": [1.],
                "dilution_factor": [0.0],
                "num_signals": [0],
                "freq_agg": ["k"]
            }
            if prob_combine_type == "weight":
                ranges["know_prob"] = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
            elif prob_combine_type in {"meta-truelearn", "meta-truelearn-greedy"}:
                ranges["tau_factor"] = [.5]
                ranges["know_prob"] = [.5]
            else:
                ranges["know_prob"] = [0.0]
        else:
            # def_var_factor = [180, 200, 300, 500]  # 1 # topic
            # def_var_factor = [25, 50, 100, 200, 250, 300, 500]  # 5 topic
            def_var_factor = [250]  # 5 topic

            # i_def_var_factor = [5, 6, 10] # 1 topic
            # i_def_var_factor = [1, 2.5, 5, 6, 10, 25, 50, 100, 250]  # 5 topic
            i_def_var_factor = [50]  # 5 topic

            ranges = {
                "def_var_factor": def_var_factor,
                "i_def_var_factor": i_def_var_factor,
                "var_constant": [0],
                "beta_factor": [.5],
                "tau_factor": [.0],
                "interest_decay_type": ["short"],
                "interest_decay_factor": [.0],
                "draw_probability": ["static"],
                "draw_factor": [1.],
                "dilution_factor": [0.0],
                "num_signals": [0],
                "freq_agg": ["k"]
            }
            if prob_combine_type == "weight":
                ranges["know_prob"] = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
            elif prob_combine_type in {"meta-truelearn", "meta-truelearn-greedy"}:
                ranges["tau_factor"] = [.0, .001, .01, .025, .05, .1, .25, .5, .75, 1.]
                ranges["know_prob"] = [0.0001, 0.001, 0.01, 0.025, 0.05, 0.1, .25, .5, .75, 1., 1.5, 5, 10]
            else:
                ranges["know_prob"] = [0.0]

    elif algorithm == "semantic_truelearn_hybrid":
        skill_repr = "cosine"

        if is_full_dataset:
            def_var_factor = [50, 100, 250, 500, 750, 1000, 1500]  # 250, 500, 1000]  # for full dataset
            i_def_var_factor = [10, 25, 50, 100, 250, 750, 1000]  # 250, 500, 1000]  # for full dataset

            ranges = {
                "def_var_factor": def_var_factor,
                "i_def_var_factor": i_def_var_factor,
                "var_constant": [0],
                "beta_factor": [.5],
                "tau_factor": [0.],
                "interest_decay_type": ["short"],
                "interest_decay_factor": [.0],
                "draw_probability": ["static"],
                "draw_factor": [1.],
                "dilution_factor": [0.0],
                "num_signals": [0],
                "freq_agg": ["k"]
            }
            if prob_combine_type == "weight":
                ranges["know_prob"] = [.1, 0.25, .5, .75, .9]
            else:
                ranges["know_prob"] = [0.0]

        else:
            def_var_factor = [50, 75, 80, 100, 200]  # for top 20
            i_def_var_factor = [1, 3, 7, 10, 15, 20]

            ranges = {
                "def_var_factor": def_var_factor,
                "i_def_var_factor": i_def_var_factor,
                "var_constant": [0],
                "beta_factor": [.5],
                "tau_factor": [0.],
                "interest_decay_type": ["short"],
                "interest_decay_factor": [.0],
                "draw_probability": ["static"],
                "draw_factor": [1.],
                "dilution_factor": [0.0],
                "num_signals": [0],
                "freq_agg": ["k"]
            }
            if prob_combine_type == "weight":
                ranges["know_prob"] = [.1, 0.25, .5, .75, .9]
            else:
                ranges["know_prob"] = [0.0]

    elif algorithm in {"truelearn_qink_pop_pred", "truelearn_qink_weighted"}:
        skill_repr = "cosine"

        if is_full_dataset:
            # def_var_factor = [250, 500, 750]  # 1 topic
            # def_var_factor = [5000, 10000, 25000, 50000]  # 3 topics
            def_var_factor = [5000]

            # i_def_var_factor = [250, 500, 1000] # 1 topic
            # i_def_var_factor = [7, 10, 25, 50, 75, 100]  # 5 topic
            i_def_var_factor = [250]

            ranges = {
                "def_var_factor": def_var_factor,
                "i_def_var_factor": i_def_var_factor,
                "var_constant": [0],
                "beta_factor": [.5],
                "tau_factor": [.0],
                "interest_decay_type": ["short"],
                "interest_decay_factor": [.0],
                "draw_probability": ["static"],
                "draw_factor": [1.],
                "dilution_factor": [0.0],
                # "num_signals": [0],
                "num_signals": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "freq_agg": ["n_vid"],
                # "freq_agg": ["n_events"],
            }
            if prob_combine_type == "weight":
                ranges["know_prob"] = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
            elif prob_combine_type in {"meta-truelearn", "meta-truelearn-greedy"}:
                ranges["tau_factor"] = [.5]
                ranges["know_prob"] = [.5]
            else:
                ranges["know_prob"] = [0.0]
        else:
            # def_var_factor = [180, 200, 300, 500]  # 1 # topic
            # def_var_factor = [25, 50, 100, 200, 250, 300, 500]  # 5 topic
            def_var_factor = [250]  # 5 topic

            # i_def_var_factor = [5, 6, 10] # 1 topic
            # i_def_var_factor = [1, 2.5, 5, 6, 10, 25, 50, 100, 250]  # 5 topic
            i_def_var_factor = [50]  # 5 topic

            ranges = {
                "def_var_factor": def_var_factor,
                "i_def_var_factor": i_def_var_factor,
                "var_constant": [0],
                "beta_factor": [.5],
                "tau_factor": [.0],
                "interest_decay_type": ["short"],
                "interest_decay_factor": [.0],
                "draw_probability": ["static"],
                "draw_factor": [1.],
                "dilution_factor": [0.0],
                "num_signals": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "freq_agg": ["n_vid"],
                # "freq_agg": ["n_events"],
            }
            if prob_combine_type == "weight":
                ranges["know_prob"] = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
            elif prob_combine_type in {"meta-truelearn", "meta-truelearn-greedy"}:
                ranges["tau_factor"] = [0.375]
                ranges["know_prob"] = [.5]
            else:
                ranges["know_prob"] = [0.0]

    # generate combos
    cnt = 0
    combos = {}

    if dilute_var:
        dilution_factors = ranges["dilution_factor"]
    else:
        dilution_factors = [0.]

    if is_final:
        ranges["def_var_factor"] = _final_def_var[(algorithm, sr_func, pred_only, is_full_dataset)]

    if algorithm in {"cbf", 'ccf', "jaccard", "user_interest"}:
        thresholds = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    else:
        thresholds = [.5]

    if algorithm == "truelearn_qink_weighted":
        ranges["tau_factor"] = [.35, .375, .4, .45, .5, .75]
        ranges["know_prob"] = [.3, .4, .5, .6, .7]
        ranges["num_signals"] = [1]

    for def_var in ranges["def_var_factor"]:
        for i_def_var in ranges.get("i_def_var_factor", [0.]):
            for know_prob in ranges.get("know_prob", [0.]):
                for var_const in ranges["var_constant"]:
                    for beta in ranges["beta_factor"]:
                        for tau in ranges["tau_factor"]:
                            _tau = know_prob * tau  # only for meta-truelearn
                            for decay_type in ranges["interest_decay_type"]:
                                for decay in ranges["interest_decay_factor"]:
                                    for top_k_sr in [top_k_sr_topics]:
                                        for dil in dilution_factors:
                                            for thresh in thresholds:
                                                for draw_p in ranges["draw_probability"]:
                                                    for draw_f in ranges["draw_factor"]:
                                                        for qual_agg in ranges["freq_agg"]:
                                                            for qual_n in ranges["num_signals"]:

                                                                # if draw probability is static, you set it and move on.
                                                                if draw_p == "static" and draw_f != 1.:
                                                                    continue

                                                                combos[cnt] = {
                                                                    "skill_repr": skill_repr,
                                                                    "algorithm": algorithm,
                                                                    "def_var_factor": def_var,
                                                                    "i_def_var_factor": i_def_var,
                                                                    "beta_factor": beta,
                                                                    "tau_factor": _tau,
                                                                    "know_prob": know_prob,
                                                                    "interest_decay_type": decay_type,
                                                                    "interest_decay_factor": decay,
                                                                    "draw_probability": draw_p,
                                                                    "draw_factor": draw_f,
                                                                    "threshold": thresh,
                                                                    "num_topics": top_n,
                                                                    "positive_only": pos_only,
                                                                    "dilute_var": dilute_var,
                                                                    "dilution_factor": dil,
                                                                    "var_constant": var_const,
                                                                    "top_k_sr_topics": top_k_sr,
                                                                    "full_dataset": is_full_dataset,
                                                                    "time": is_timing,
                                                                    "topics": topics,
                                                                    "has_part_id": True,
                                                                    "prob_combine_type": prob_combine_type,
                                                                    "q_random":q_random
                                                                }
                                                                if "semantic" in algorithm:
                                                                    combos[cnt][
                                                                        "semantic_relatedness_filepath"] = sem_path
                                                                    combos[cnt]["agg_func"] = agg_func
                                                                    combos[cnt]["prediction_only"] = pred_only
                                                                    combos[cnt]["sr_func"] = sr_func

                                                                if algorithm in {"ccf", "user_tfidf"}:
                                                                    combos[cnt]["source_filepath"] = source

                                                                if algorithm in {"truelearn_qink_pop_pred",
                                                                                 "truelearn_qink_weighted",
                                                                                 "truelearn_novelq_pop_pred"}:
                                                                    combos[cnt][
                                                                        "quality_mapping_filepath"] = quality_path
                                                                    combos[cnt]["freq_type"] = quality_type
                                                                    combos[cnt]["num_signals"] = qual_n
                                                                    combos[cnt]["freq_agg"] = qual_agg

                                                                cnt += 1

    return combos


def normalise_item(item):
    return {
        "account_id": item["session"],
        "accuracy": item["accuracy"],
        "precision": item["precision"],
        "recall": item["recall"],
        "f1": item["f1"],
        "roc_score": item["roc_score"],
        "pr_score": item["pr_score"],

        "num_events": item["num_events"],
        "topic_sparsity_rate": item["num_topics_rate"]  # num unique topics / num events
    }


def get_summary_result(result_df):
    """calculate accuracy, prec, recall, f1 and weighted versions

    Args:
        result_df (DataFrame):

    Returns:
        {str: val}
    """
    stat_dict = {}

    roc_df = result_df[result_df["roc_score"] >= 0.]  # subset of records for roc auc calculation
    pr_def = result_df[result_df["pr_score"] >= 0.]  # subset of records for pr auc calculation

    stat_dict["accuracy"] = np.average(result_df["accuracy"])
    stat_dict["precision"] = np.average(result_df["precision"])
    stat_dict["recall"] = np.average(result_df["recall"])
    stat_dict["f1"] = np.average(result_df["f1"])
    stat_dict["roc_score"] = np.average(roc_df["roc_score"])
    stat_dict["pr_score"] = np.average(pr_def["pr_score"])

    stat_dict["accuracy_w"] = np.average(result_df["accuracy"], weights=result_df["num_events"])
    stat_dict["precision_w"] = np.average(result_df["precision"], weights=result_df["num_events"])
    stat_dict["recall_w"] = np.average(result_df["recall"], weights=result_df["num_events"])
    stat_dict["f1_w"] = np.average(result_df["f1"], weights=result_df["num_events"])
    stat_dict["roc_score_w"] = np.average(roc_df["roc_score"], weights=roc_df["num_events"])
    stat_dict["pr_score_w"] = np.average(pr_def["pr_score"], weights=pr_def["num_events"])

    return stat_dict


def get_result_mapping(path):
    results_df = pd.read_csv(path)
    results = results_df.to_dict(orient='records')

    return results


def main(args):
    from analyses.truelearn_experiments.run_experiments import main as tl_main

    if args["positive_only"]:
        algo_folder_name = args["algorithm"] + "_" + "pos"
    else:
        algo_folder_name = args["algorithm"] + "_" + "neg"

    if args["algorithm"].startswith("semantic"):
        algo_folder_name += "_" + args["agg_func"]

    if args["predict_only"]:
        algo_folder_name += "_pred"

    if args["dilute_var"]:
        algo_folder_name += "_dil"

    if args["sr_func"] == "pr":
        algo_folder_name += "_pr"

    if args["sr_func"] == "gauss":
        algo_folder_name += "_gauss"

    algo_folder_name += "_{}".format(args["freq_type"])

    # generate the required folders
    output_folder = join(args["output_dir"], algo_folder_name)

    try:
        makedirs(output_folder)
    except FileExistsError:
        print("Folder already exists. Overwriting results")

    # input training data paths
    validation_datapath = join(args["dataset_dir"], "session_data_validation.csv")
    test_datapath = join(args["dataset_dir"], "session_data_test.csv")

    # generate the results folder
    test_result_folder = join(output_folder, "_test")

    try:
        makedirs(test_result_folder)
    except FileExistsError:
        print("Folder already exists. Overwriting results")

    # ------ validation phase ------
    # generate grid search hyperparameters

    results = []  # to store results

    hyperparameters = generate_hyperparameters(args["algorithm"], top_n=args["num_topics"],
                                               pos_only=args["positive_only"],
                                               sem_path=args["semantic_relatedness_filepath"],
                                               pred_only=args["predict_only"], agg_func=args["agg_func"],
                                               dilute_var=args["dilute_var"], top_k_sr_topics=args["top_k_sr_topics"],
                                               sr_func=args["sr_func"], is_full_dataset=args["full_dataset"],
                                               is_final=args["is_final"], is_timing=args["is_timing"],
                                               topics=args["topics"], prob_combine_type=args["prob_combine_type"],
                                               source=args["source_filepath"],
                                               quality_path=args["quality_mapping_filepath"],
                                               quality_type=args["freq_type"], q_random=args["q_random"])

    is_validation = True
    if args["algorithm"] == 'persistent' or args["algorithm"] == 'majority':
        is_validation = False

    top_hyperparameter_combo = hyperparameters[0]
    top_hyperparameter_combo["dataset_filepath"] = test_datapath
    top_hyperparameter_combo["output_dir"] = test_result_folder

    if is_validation:
        # for each combo, run the experiment with the hyperparameters on validation set
        for run_id, tmp_args in hyperparameters.items():
            # generate the results folder
            results_folder = join(output_folder, str(run_id))
            print("Creating {}".format(join(output_folder, str(run_id))))
            try:
                makedirs(results_folder)
            except FileExistsError:
                if args["resume_job"]:
                    print("Folder already exists. Going to next hyper-parameter configuration")
                    results = get_result_mapping(join(test_result_folder, "validation_metrics.csv"))
                    continue
                print("Folder already exists. Overwriting results")

            tmp_args["dataset_filepath"] = validation_datapath
            tmp_args["output_dir"] = results_folder
            tmp_args["n_jobs"] = args["n_jobs"]

            try:
                tmp_result = tl_main(tmp_args)
            except Exception as e:
                print(tmp_args)
                raise e

            tmp_result = pd.DataFrame([normalise_item(itm) for itm in tmp_result])

            # first remove the learners that only has zero class
            # tmp_result = tmp_result[tmp_result["roc_score"] >= 0.]
            # tmp_result = tmp_result[tmp_result["pr_score"] >= 0.]

            # summarise result
            summary_result = get_summary_result(tmp_result)
            summary_result["run_id"] = run_id

            results.append(summary_result)

            if len(results) != 0:
                validation_result = pd.DataFrame(results)[
                    ["run_id", "accuracy", "precision", "recall", "f1", "roc_score", "pr_score",
                     "accuracy_w", "precision_w", "recall_w", "f1_w", "roc_score_w", "pr_score_w"]]
                validation_result.to_csv(join(test_result_folder, "validation_metrics.csv"), index=False)

            # find the best result hyperparameter combo
            # results.sort(key=lambda l: - l["f1_w"])  # sort by biggest f1
            # results.sort(key=lambda l: - l["pr_score_w"])  # sort by biggest f1

    results.sort(key=lambda l: - l[args["ranking_metric"]])  # sort by biggest prec-recall auc
    top_id = results[0]["run_id"]

    # run test set with the best combo
    top_hyperparameter_combo = hyperparameters[top_id]
    top_hyperparameter_combo["dataset_filepath"] = test_datapath
    top_hyperparameter_combo["output_dir"] = test_result_folder
    top_hyperparameter_combo["n_jobs"] = args["n_jobs"]

    # record results
    tmp_result = tl_main(top_hyperparameter_combo)

    tmp_result = pd.DataFrame([normalise_item(itm) for itm in tmp_result])

    # first remove the learners that only has zero class
    # tmp_result = tmp_result[tmp_result["roc_score"] >= 0.]
    # tmp_result = tmp_result[tmp_result["pr_score"] >= 0.]

    # summarise results
    if len(results) != 0:
        validation_result = pd.DataFrame(results)[
            ["run_id", "accuracy", "precision", "recall", "f1", "roc_score", "pr_score",
             "accuracy_w", "precision_w", "recall_w", "f1_w", "roc_score_w", "pr_score_w"]]

        validation_result.to_csv(join(test_result_folder, "validation_metrics.csv"), index=False)

    summary_result = pd.DataFrame([get_summary_result(tmp_result)])[
        ["accuracy", "precision", "recall", "f1", "roc_score", "pr_score",
         "accuracy_w", "precision_w", "recall_w", "f1_w", "roc_score_w", "pr_score_w"]]

    top_hyperparameter_combo = pd.DataFrame([top_hyperparameter_combo])

    # write results to disc
    tmp_result.to_csv(join(test_result_folder, "userwise_metrics.csv"), index=False)
    summary_result.to_csv(join(test_result_folder, "summary_metrics.csv"), index=False)

    top_hyperparameter_combo.to_csv(join(test_result_folder, "best_hyperparameters.csv"), index=False)


if __name__ == '__main__':
    """this script takes in the wikified lectures file and the learner activity data from videolectures to build a .
    output of this script will be {slug, vid_id, part_id, start_time, stop_time, clean, text, wiki_concepts}
    eg: command to run this script:

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-dir', type=str, required=True,
                        help="the directory where validation and testing data is")
    parser.add_argument('--algorithm', default='trueknowledge_sum', const='all', nargs='?',
                        choices=['engage', 'persistent', 'majority',
                                 "cbf", "ccf", "jaccard", "user_interest_cos", "user_interest_bin", "user_tfidf",
                                 "knowledge_tracing", "knowledge_tracing_interest",
                                 "truelearn_fixed", "truelearn_novel", "truelearn_interest", "truelearn_hybrid",
                                 "truelearn_novelq_pop_pred", "truelearn_qink_pop_pred", "truelearn_qink_weighted",
                                 "semantic_truelearn_fixed", "semantic_truelearn_novel",
                                 "semantic_truelearn_interest", "semantic_truelearn_hybrid"],
                        help="The name of the algorithm can be one of the allowed algorithms")
    parser.add_argument("--num-topics", type=int, default=1,
                        help="The number of top ranked topics that have to be considered.")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Output directory path where the results will be saved.")
    parser.add_argument('--semantic-relatedness-filepath', type=str, default=None,
                        help="where semantic relatedness mapping is")
    parser.add_argument('--source-filepath', type=str, default=None,
                        help="where collabarative filtering data is")
    parser.add_argument('--positive-only', action='store_true', help="learns from positive examples only")
    parser.add_argument('--predict-only', action='store_true', help="only use semantic relatedness in prediction")
    parser.add_argument('--agg-func', default='or', const='all', nargs='?',
                        choices=['raw', 'max', 'or'],
                        help="The name of the SR aggregation method be one of the allowed methods")
    parser.add_argument("--top-k-sr-topics", type=int, default=-1,
                        help="The number of top ranked topics that have to be considered for semantic relatedness.")
    parser.add_argument('--dilute-var', action='store_true', help="dilute variance")
    parser.add_argument('--sr-func', default='raw', const='all', nargs='?', choices=['raw', 'pr', "gauss"],
                        help="What SR aggregation method is to be used")
    parser.add_argument('--n-jobs', type=str, default="*",
                        help="number of parallel jobs")
    parser.add_argument('--full-dataset', action='store_true', help="if full dataset or smaller dataset.")
    parser.add_argument('--resume-job', action='store_true', help="resuming a job broken halfway.")
    parser.add_argument('--is-final', action='store_true', help="resuming a job broken halfway.")
    parser.add_argument('--is-timing', action='store_true', help="if the jobs are being timed")
    parser.add_argument('--ranking-metric', default='f1_w', const='all', nargs='?',
                        choices=['f1_w', 'pr_score_w'],
                        help="determines which metric should be used for ranking")
    parser.add_argument('--topics', action='store_true', help="if related topics should be stored")
    parser.add_argument('--prob-combine-type', default='short', const='all', nargs='?',
                        choices=['and', "or", "weight", "acc_weight", "f1_weight", "meta-logistic", "meta-perceptron",
                                 "meta-truelearn", "meta-truelearn-greedy"],
                        help="Type of fucntion used to combine knowledge and interest")
    parser.add_argument('--quality-mapping-filepath', type=str, required=True,
                        help="mapping of engagement values")
    parser.add_argument('--freq-type', default='k', const='all', nargs='?',
                        choices=['k', 'i', 'ki'],
                        help="The name of the algorithm can be one of the allowed algorithms")
    parser.add_argument('--q-random', action='store_true', help="makes the first prediction random in prediction.")
    args = vars(parser.parse_args())

    main(args)
