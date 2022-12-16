from functools import partial
from os.path import join
import numpy as np

import json

from analyses.truelearn_experiments.semantic_truelearn.diagnostics._lib.truelearn_models import truelearn_novel_model
from analyses.truelearn_experiments.trueknowledge_recommender_models import get_semantic_relatedness_mapping, \
    get_default_variance_from_coverage_values, all_positive_tkts_team_model
from analyses.truelearn_experiments.utils import convert_to_records, vectorise_data
from lib.spark_context import get_spark_context


def _get_eval_func(algorithm, vect_type, data=None, engage_func="all", threshold=.5, def_var_factor=0.5, tau_factor=0.1,
                   beta_factor=.1, draw_probability="static", positive_only=True, draw_factor=.1,
                   interest_decay_factor=0., semantic_mapping_path=None, agg_func="raw", is_pred_only=False,
                   is_diluted=False, dil_factor=1.0):
    if algorithm == "semantic_truelearn_fixed":
        semantic_mapping = get_semantic_relatedness_mapping(semantic_mapping_path)
        spark = get_spark_context()
        semantic_mapping_b = spark.sparkContext.broadcast(semantic_mapping)
        cosine_var = float(get_default_variance_from_coverage_values(data, vect_type))
        _def_var = float(np.square(cosine_var * def_var_factor))
        _beta_sqr = float(np.square(np.sqrt(_def_var) * beta_factor))
        _tau = float(1. * tau_factor)
        return partial(all_positive_tkts_team_model, init_skill=0., def_var=_def_var, tau=_tau,
                       beta_sqr=_beta_sqr, threshold=threshold, positive_only=positive_only,
                       semantic_mapping=semantic_mapping_b, agg_func=agg_func, is_pred_only=is_pred_only)

    else:
        semantic_mapping = get_semantic_relatedness_mapping(semantic_mapping_path)
        spark = get_spark_context()
        semantic_mapping_b = spark.sparkContext.broadcast(semantic_mapping)
        cosine_var = float(get_default_variance_from_coverage_values(data, vect_type))
        _def_var = float(np.square(cosine_var * def_var_factor))
        _beta_sqr = float(np.square(np.sqrt(_def_var) * beta_factor))
        _tau = float(1. * tau_factor)
        return partial(truelearn_novel_model, init_skill=0., def_var=_def_var, tau=_tau, beta_sqr=_beta_sqr,
                       threshold=threshold, draw_probability=draw_probability, positive_only=positive_only,
                       draw_factor=draw_factor, semantic_mapping=semantic_mapping_b, agg_func=agg_func,
                       is_pred_only=is_pred_only, is_diluted=is_diluted, dil_factor=dil_factor, debug=True)


def restructure_data(line):
    (sess, (acc, prec, rec, f1, num_events, is_stats)) = line

    temp_dict = {
        "session": sess,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "num_events": num_events,
        "num_topics_rate": 0
    }

    if is_stats:
        temp_dict["positive_rate"] = is_stats["positive"]
        temp_dict["predict_positive_rate"] = is_stats["predict_positive"]
        temp_dict["change_label_rate"] = is_stats["change_label"]
        temp_dict["num_topics"] = is_stats["num_topics"]
        temp_dict["num_topics_rate"] = is_stats["num_topics_rate"]
        temp_dict["num_user_topics"] = is_stats["num_user_topics"]

        temp_dict["user_model"] = is_stats["user_model"]

    return temp_dict


def main(args):
    spark = get_spark_context(master="local[{}]".format(args["n_jobs"]))
    spark.sparkContext.setLogLevel("ERROR")

    # if dilute factor is zero, no dilution
    if args["dilution_factor"] == 0.:
        args["dilute_var"] = False

    # load training data
    data = (spark.read.csv(args["dataset_filepath"], sep=",", header=False).
            rdd.
            map(lambda l: convert_to_records(l, top_n=args["num_topics"], has_part_id=True)))

    grouped_data = data.map(lambda l: (l["session"], l)).groupByKey(numPartitions=1000).mapValues(list)

    # run the algorithm to get results
    if (args["algorithm"]) == "truelearn_fixed":
        eval_func = _get_eval_func(args["algorithm"], args["skill_repr"], data=grouped_data,
                                   def_var_factor=args["def_var_factor"], tau_factor=args["tau_factor"],
                                   beta_factor=args["beta_factor"], threshold=args["threshold"],
                                   positive_only=args["positive_only"])

    elif (args["algorithm"]) == "semantic_truelearn_fixed":
        eval_func = _get_eval_func(args["algorithm"], args["skill_repr"], data=grouped_data,
                                   def_var_factor=args["def_var_factor"], tau_factor=args["tau_factor"],
                                   beta_factor=args["beta_factor"], threshold=args["threshold"],
                                   positive_only=args["positive_only"],
                                   semantic_mapping_path=args["semantic_relatedness_filepath"],
                                   agg_func=args['sr_agg_func'], is_pred_only=args["prediction_only"])

    elif (args["algorithm"]) == "truelearn_novel":
        eval_func = _get_eval_func(args["algorithm"], args["skill_repr"], data=grouped_data,
                                   def_var_factor=args["def_var_factor"], tau_factor=args["tau_factor"],
                                   beta_factor=args["beta_factor"], threshold=args["threshold"],
                                   draw_probability=args["draw_probability"], draw_factor=args["draw_factor"],
                                   positive_only=False)

    elif (args["algorithm"]) == "semantic_truelearn_novel":
        eval_func = _get_eval_func(args["algorithm"], args["skill_repr"], data=grouped_data,
                                   def_var_factor=args["def_var_factor"], tau_factor=args["tau_factor"],
                                   beta_factor=args["beta_factor"], threshold=args["threshold"],
                                   draw_probability=args["draw_probability"], draw_factor=args["draw_factor"],
                                   positive_only=False, semantic_mapping_path=args["semantic_relatedness_filepath"],
                                   agg_func=args['sr_agg_func'], is_pred_only=args["prediction_only"],
                                   is_diluted=args["dilute_var"], dil_factor=args["dilution_factor"])

    elif (args["algorithm"]) == "trueknowledge_all":
        eval_func = _get_eval_func(args["algorithm"], args["skill_repr"], data=grouped_data)
    elif (args["algorithm"]) == "truelearn_background":
        eval_func = _get_eval_func(args["algorithm"], args["skill_repr"], def_var_factor=args["def_var_factor"],
                                   tau_factor=args["tau_factor"], beta_factor=args["beta_factor"],
                                   threshold=args["threshold"], positive_only=args["positive_only"])

    else:
        eval_func = _get_eval_func(args["algorithm"], args["skill_repr"])

    # test = grouped_data.mapValues(lambda l: vectorise_data(l, args["skill_repr"])).filter(
    #     lambda l: l[0] == "15720").first()
    #
    # _ = eval_func(test[1])

    vectorised_data = grouped_data.mapValues(lambda l: vectorise_data(l, args["skill_repr"]))

    # test = vectorised_data.first()
    #
    # _ = eval_func(test[1])

    # test = vectorised_data.collect()
    # for id, events in test:
    #     print("user {} started!!!".format(id))
    #     eval_func(events)
    #
    #
    # import sys
    # sys.exit()

    evaluated_data = vectorised_data.mapValues(eval_func)

    # if args["algorithm"] == "truelearn_background":
    #     # run a for-loop
    #     # temp_data = evaluated_data.collect()
    #     # evaluated_data = []
    #     # for (user, events) in temp_data:
    #     #     try:
    #     #         evaluated_data.append((user, truelearn_background_model(events, def_var=float(args["def_var_factor"]),
    #     #                                                                 tau=float(args["tau_factor"]),
    #     #                                                                 beta_sqr=float(args["beta_factor"]),
    #     #                                                                 threshold=float(args["threshold"]),
    #     #                                                                 positive_only=args["positive_only"])))
    #     #     except ValueError:
    #     #         print()
    #
    #     evaluated_data = spark.sparkContext.parallelize(evaluated_data, 10)
    #
    # else:
    #     evaluated_data = evaluated_data.mapValues(eval_func)

    restructured_data = evaluated_data.map(restructure_data).collect()

    with open(join(args["output_dir"], "model_results.json"), "w") as outfile:
        json.dump(restructured_data, outfile)

    return restructured_data


if __name__ == '__main__':
    """this script takes in the wikified lectures file and the learner activity data from videolectures to build a .
    output of this script will be {slug, vid_id, part_id, start_time, stop_time, clean, text, wiki_concepts}
    eg: command to run this script:

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-filepath', type=str, required=True,
                        help="where training data is")
    parser.add_argument('--semantic-relatedness-filepath', type=str, required=True,
                        help="where training data is")
    parser.add_argument('--sr-agg-func', default='max', const='all', nargs='?',
                        choices=['raw', 'max', 'or'],
                        help="The name of the SR aggregation method be one of the allowed methods")
    parser.add_argument('--algorithm', default='trueknowledge_sum', const='all', nargs='?',
                        choices=['engage', 'persistent', 'majority', "truelearn_novel", "truelearn_fixed",
                                 "truelearn_background", "semantic_truelearn_novel", "semantic_truelearn_fixed"],
                        help="The name of the algorithm can be one of the allowed algorithms")
    parser.add_argument("--num-topics", type=int, default=10,
                        help="The number of top ranked topics that have to be considered.")
    parser.add_argument('--skill-repr', default='cosine', const='all', nargs='?',
                        choices=['cosine', 'binary', "norm"],
                        help="How the skills should be represented in the models")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Output directory path where the results will be saved.")
    parser.add_argument('--engage-func', default='all', const='all', nargs='?',
                        choices=['all', 'sum', "quality"],
                        help="What engagement eval function to be used")
    parser.add_argument('--threshold', type=float, default=1.,
                        help="Probability threshold for classifying true")
    parser.add_argument('--def-var-factor', type=float, default=.5,
                        help="Probability of knowing this topics")
    parser.add_argument('--tau-factor', type=float, default=.0,
                        help="Probability of watching even when cant learn")
    parser.add_argument('--interest-decay-factor', type=float, default=.0,
                        help="Probability of watching even when cant learn")
    parser.add_argument('--beta-factor', type=float, default=.1,
                        help="Probability skipping even when can learn")
    parser.add_argument('--draw-probability', type=str, default="static",
                        help="Probability of drawing the match")
    parser.add_argument('--draw-factor', type=float, default=.1,
                        help="factor of draw probability to be used")
    parser.add_argument('--positive-only', action='store_true', help="learns from negative examples too")
    parser.add_argument('--prediction-only', action='store_true',
                        help="semantic relatedness is only considered at prediction time")
    parser.add_argument('--dilute-var', action='store_true', help="dilute variance")
    parser.add_argument('--dilution-factor', type=float, default=.0,
                        help="factor variance dilution to be enforced")
    parser.add_argument('--is-video', action='store_true', help="if the prediction is done on full video or not")
    parser.add_argument('--n-jobs', type=str, default="*",
                        help="number of parallel jobs")

    args = vars(parser.parse_args())

    _ = main(args)
