from functools import partial
from os.path import join
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning

from analyses.truelearn_experiments.baseline_models import content_based_filtering_model, jaccard_based_filtering_model, \
    content_collaborative_filtering_model, user_interest_model, user_interest_tfidf_model
from analyses.truelearn_experiments.baseline_recommender_models import persistent_model, majority_model, engage_model
import json

from analyses.truelearn_experiments.hybrid_model import hybrid_truelearn_model, qink_truelearn_model, \
    get_quality_mapping, truelearn_novelq_model
from analyses.truelearn_experiments.semantic_truelearn.data_preperation.generate_concept_frequencies import \
    get_document_frequency_mapping
from analyses.truelearn_experiments.semantic_truelearn.data_preperation.generate_lecture_transitions_mapping import \
    get_lecture_transition
from analyses.truelearn_experiments.trueknowledge_recommender_models import get_default_variance_from_coverage_values, \
    truelearn_novel_model, get_interest_decay_func

from analyses.truelearn_experiments.knowledge_tracing_model import knowledge_tracing_model
from analyses.truelearn_experiments.utils import convert_to_records, vectorise_data, get_semantic_relatedness_mapping
from lib.spark_context import get_spark_context

import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

MIN_NUM_EVENTS = 5


def _get_eval_func(algorithm, vect_type, data=None, engage_func="all", threshold=.5, def_var_factor=0.5,
                   i_def_var_factor=0.5, tau_factor=0.1, beta_factor=.1, draw_probability="static", positive_only=True,
                   draw_factor=.1, interest_decay_type=None, interest_decay_factor=0., semantic_mapping_path=None,
                   agg_func="raw", is_pred_only=False, is_diluted=False, dil_factor=1.0, var_const=0.,
                   top_k_sr_topics=1, sr_func="raw", is_timing=False, is_topics=False,
                   prob_combine_type="weight", know_prob=1., source=None, start_event=0, quality_mapping=None,
                   num_signals=0, freq_type="k", freq_agg="sum", q_random=False):
    if algorithm == "engage":
        return engage_model
    elif algorithm == "persistent":
        return persistent_model
    elif algorithm == "majority":
        return majority_model

    elif algorithm == "cbf":
        return partial(content_based_filtering_model, threshold=threshold)
    elif algorithm == "ccf":
        transition_mapping = get_lecture_transition(source)
        spark = get_spark_context()
        transition_mapping_b = spark.sparkContext.broadcast(transition_mapping)
        return partial(content_collaborative_filtering_model, transition_mapping=transition_mapping_b,
                       threshold=threshold, start_event=start_event)
    elif algorithm == "jaccard":
        return partial(jaccard_based_filtering_model, threshold=threshold, start_event=start_event)
    elif algorithm == "user_interest":
        return partial(user_interest_model, threshold=threshold, start_event=start_event)
    elif algorithm == "user_tfidf":
        df_mapping = get_document_frequency_mapping(source)
        spark = get_spark_context()
        df_mapping_b = spark.sparkContext.broadcast(df_mapping)
        return partial(user_interest_tfidf_model, threshold=threshold, df_mapping=df_mapping_b, start_event=start_event)

    elif algorithm == "truelearn_fixed":
        # these normal floatations are MANDATORY (-_____-)
        cosine_var = float(get_default_variance_from_coverage_values(data, vect_type))
        _def_var = float(np.square(cosine_var * def_var_factor))
        _beta_sqr = float(np.square(np.sqrt(_def_var) * beta_factor))
        _tau = float(1. * tau_factor)
        return partial(truelearn_novel_model, init_skill=0., def_var=_def_var, tau=_tau,
                       beta_sqr=_beta_sqr, threshold=threshold, positive_only=positive_only, is_fixed=True,
                       start_event=start_event)


    elif algorithm == "truelearn_novel":
        cosine_var = float(get_default_variance_from_coverage_values(data, vect_type))
        _def_var = float(np.square(cosine_var * def_var_factor))
        _beta_sqr = float(np.square(np.sqrt(_def_var) * beta_factor))
        _tau = float(1. * tau_factor)
        return partial(truelearn_novel_model, init_skill=0., def_var=_def_var, tau=_tau, beta_sqr=_beta_sqr,
                       threshold=threshold, draw_probability=draw_probability, positive_only=positive_only,
                       draw_factor=draw_factor, var_const=var_const, tracking=is_timing, is_topics=is_topics,
                       start_event=start_event)

    elif algorithm == "truelearn_novelq_pop_pred":
        cosine_var = float(get_default_variance_from_coverage_values(data, vect_type))
        _def_var = float(np.square(cosine_var * def_var_factor))
        _beta_sqr = float(np.square(np.sqrt(_def_var) * beta_factor))

        _tau = float(1. * tau_factor)

        _quality_mapping = get_quality_mapping(quality_mapping)

        spark = get_spark_context()
        quality_mapping_b = spark.sparkContext.broadcast(_quality_mapping)

        return partial(truelearn_novelq_model, init_skill=0., k_def_var=_def_var, tau=_tau, k_beta_sqr=_beta_sqr,
                       threshold=threshold, draw_probability=draw_probability, positive_only=positive_only,
                       draw_factor=draw_factor, var_const=var_const, tracking=is_timing, is_topics=is_topics,
                       k_topics=2, start_event=start_event, quality_mapping=quality_mapping_b,
                       quality_type="prediction", num_signals=num_signals, freq_type=freq_type, freq_agg=freq_agg)


    elif algorithm == "truelearn_interest":
        cosine_var = float(get_default_variance_from_coverage_values(data, vect_type))
        _def_var = float(np.square(cosine_var * def_var_factor))
        _beta_sqr = float(np.square(np.sqrt(_def_var) * beta_factor))
        _tau = float(1. * tau_factor)
        interest_decay_func = get_interest_decay_func(interest_decay_type, interest_decay_factor)
        return partial(truelearn_novel_model, init_skill=0., def_var=_def_var, tau=_tau, beta_sqr=_beta_sqr,
                       threshold=threshold, draw_probability=draw_probability, positive_only=positive_only,
                       draw_factor=draw_factor, var_const=var_const, tracking=is_timing, is_topics=is_topics,
                       is_interest=True, decay_func=interest_decay_func, start_event=start_event)


    elif algorithm == "truelearn_hybrid":
        interest_decay_func = get_interest_decay_func(interest_decay_type, interest_decay_factor)

        k_cosine_var = float(get_default_variance_from_coverage_values(data, "cosine"))
        i_cosine_var = float(get_default_variance_from_coverage_values(data, "binary"))  # for Gaussian interest

        _k_def_var = float(np.square(k_cosine_var * def_var_factor))
        # _i_def_var = float(i_def_var_factor)  # for init unceratinty for kt interest
        _i_def_var = float(np.square(i_cosine_var * i_def_var_factor))  # for Gaussian interest

        _k_beta_sqr = float(np.square(np.sqrt(_k_def_var) * beta_factor))
        # _i_beta_sqr = float(beta_factor)  # for kt pfail
        _i_beta_sqr = float(np.square(np.sqrt(_i_def_var) * beta_factor))  # for Gaussian interest

        _tau = float(1. * tau_factor)  # for pguess

        return partial(hybrid_truelearn_model, init_skill=0., k_def_var=_k_def_var, i_def_var=_i_def_var,
                       tau=_tau, k_beta_sqr=_k_beta_sqr, i_beta_sqr=_i_beta_sqr, threshold=threshold,
                       draw_probability=draw_probability, positive_only=positive_only, draw_factor=draw_factor,
                       var_const=var_const, tracking=is_timing, is_topics=is_topics, decay_func=interest_decay_func,
                       prob_combine_type=prob_combine_type, know_prob=know_prob, k_topics=3, i_topics=5,
                       start_event=start_event, q_random=q_random)


    elif algorithm == "knowledge_tracing_interest":
        _init_certainty = float(def_var_factor)  # .0
        _beta_sqr = float(beta_factor)  # pfail
        _tau = float(tau_factor)  # pguess
        interest_decay_func = get_interest_decay_func(interest_decay_type, interest_decay_factor)
        temp_func = partial(knowledge_tracing_model, init_certainty=_init_certainty, tau=_tau, beta_sqr=_beta_sqr,
                            threshold=threshold, positive_only=False, is_interest=True, decay_func=None,
                            start_event=start_event)
        return temp_func

    elif algorithm == "knowledge_tracing":
        _init_certainty = float(def_var_factor)  # .0
        _beta_sqr = float(beta_factor)  # pfail
        _tau = float(tau_factor)  # pguess
        temp_func = partial(knowledge_tracing_model, init_certainty=_init_certainty, tau=_tau, beta_sqr=_beta_sqr,
                            threshold=threshold, positive_only=positive_only, start_event=start_event)
        return temp_func


def restructure_data(line):
    # accuracy, precision, recall, f1, roc_score, pr_score, int(num_records), stats
    (sess, (acc, prec, rec, f1, roc_score, pr_score, num_events, is_stats)) = line

    temp_dict = {
        "session": sess,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_score": roc_score,
        "pr_score": pr_score,
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
        temp_dict["total_duration"] = is_stats.get("total_duration", 0)
        temp_dict["rel_topics"] = is_stats.get("rel_topics")

        temp_dict["num_updates"] = is_stats.get("num_updates", [])
        temp_dict["topics"] = is_stats.get("topics", [])

        temp_dict["user_model"] = is_stats["user_model"]

        temp_dict["actual"] = is_stats["actual"]
        temp_dict["predicted"] = is_stats["predicted"]

        temp_dict["vid_changes"] = is_stats.get("vid_changes", 0)
        temp_dict["vid_pos_changes"] = is_stats.get("vid_pos_changes", 0)
        temp_dict["video_change"] = is_stats.get("video_change", [])

        if is_stats["is_hybrid"]:
            temp_dict["i_num_topics_rate"] = is_stats["i_num_topics_rate"]
            temp_dict["i_num_user_topics"] = is_stats["i_num_user_topics"]

            temp_dict["k_probs"] = is_stats["k_preds"]
            temp_dict["i_probs"] = is_stats["i_preds"]
            temp_dict["weights"] = is_stats["weights"]

            temp_dict["i_num_updates"] = is_stats.get("i_num_updates", [])
            temp_dict["i_topics"] = is_stats.get("i_topics", [])

    return temp_dict


def main(args):
    if args["full_dataset"]:
        NUM_PARTITIONS = 1000
    else:
        NUM_PARTITIONS = 20

    spark = get_spark_context(master="local[{}]".format(args["n_jobs"]), driver_memroy="30g", exectutor_memory="30g",
                              max_result_size="4g")
    spark.sparkContext.setLogLevel("ERROR")

    is_debug = args.get("debug", False)  # debug mode

    # if dilute factor is zero, no dilution
    if args["dilution_factor"] == 0.:
        args["dilute_var"] = False

    # load training data
    data = (spark.read.csv(args["dataset_filepath"], sep=",", header=False).
            rdd.
            # map(lambda l: convert_to_records(l, top_n=args["num_topics"], has_part_id=args["has_part_id"])))
            map(lambda l: convert_to_records(l, top_n=args["num_topics"], has_part_id=True)))

    # data2 = (spark.read.csv(join(args["dataset_filepath"], "session_data_validation.csv"), sep=",", header=False).
    #         rdd.
    #         map(lambda l: convert_to_records(l, top_n=args["num_topics"], has_part_id=True)))
    #
    # data = data1.union(data2)

    grouped_data = (data.map(lambda l: (l["session"], l)).
                    groupByKey(numPartitions=NUM_PARTITIONS).
                    mapValues(list).
                    filter(lambda l: len(l[1]) >= MIN_NUM_EVENTS).
                    # filter(lambda l: any([i["label"] == 1 for i in l[1]])).
                    repartition(NUM_PARTITIONS))

    # count_learners = grouped_data.count()
    # print("using {} learners".format(count_learners))

    # run the algorithm to get results

    if (args["algorithm"]) == "truelearn_fixed":
        eval_func = _get_eval_func(args["algorithm"], args["skill_repr"], data=grouped_data,
                                   def_var_factor=args["def_var_factor"], tau_factor=args["tau_factor"],
                                   beta_factor=args["beta_factor"], threshold=args["threshold"],
                                   positive_only=args["positive_only"])


    elif (args["algorithm"]) == "truelearn_novel":
        eval_func = _get_eval_func(args["algorithm"], args["skill_repr"], data=grouped_data,
                                   def_var_factor=args["def_var_factor"], tau_factor=args["tau_factor"],
                                   beta_factor=args["beta_factor"], threshold=args["threshold"],
                                   draw_probability=args["draw_probability"], draw_factor=args["draw_factor"],
                                   var_const=args["var_constant"], positive_only=False, is_timing=args["time"],
                                   is_topics=args["topics"])


    elif (args["algorithm"]) == "truelearn_interest":
        eval_func = _get_eval_func(args["algorithm"], args["skill_repr"], data=grouped_data,
                                   def_var_factor=args["def_var_factor"], tau_factor=args["tau_factor"],
                                   beta_factor=args["beta_factor"], threshold=args["threshold"],
                                   draw_probability=args["draw_probability"], draw_factor=args["draw_factor"],
                                   var_const=args["var_constant"], positive_only=False, is_timing=args["time"],
                                   is_topics=args["topics"], interest_decay_type=args["interest_decay_type"],
                                   interest_decay_factor=args["interest_decay_factor"])


    elif (args["algorithm"]) == "truelearn_hybrid":
        eval_func = _get_eval_func(args["algorithm"], args["skill_repr"], data=grouped_data,
                                   def_var_factor=args["def_var_factor"], i_def_var_factor=args["i_def_var_factor"],
                                   tau_factor=args["tau_factor"], beta_factor=args["beta_factor"],
                                   threshold=args["threshold"], draw_probability=args["draw_probability"],
                                   draw_factor=args["draw_factor"], var_const=args["var_constant"], positive_only=False,
                                   is_timing=args["time"], is_topics=args["topics"],
                                   interest_decay_type=args["interest_decay_type"],
                                   interest_decay_factor=args["interest_decay_factor"],
                                   prob_combine_type=args["prob_combine_type"], know_prob=args["know_prob"],
                                   q_random=args["q_random"])

    elif (args["algorithm"]) == "trueknowledge_all":
        eval_func = _get_eval_func(args["algorithm"], args["skill_repr"], data=grouped_data)

    else:
        eval_func = _get_eval_func(args["algorithm"], args["skill_repr"])

    # test = grouped_data.mapValues(lambda l: vectorise_data(l, args["skill_repr"])).filter(
    #     lambda l: l[0] == "15720").first()
    #
    # _ = eval_func(test[1])

    vectorised_data = grouped_data.mapValues(lambda l: vectorise_data(l, args["skill_repr"]))

    if is_debug:

        test = vectorised_data.collect()

        results = []
        for id, events in test:
            # if id != '16':
            #     continue

            print("user {} started!!!".format(id))
            actual = eval_func(events)
            # user_model = actual[7]["user_model"]
            #
            # if is_topic_eligible(user_model.keys()):
            #     # semantic_mapping = get_semantic_relatedness_mapping(args["semantic_relatedness_filepath"])
            #     do_graph_analysis(user_model, None)
            #

            results.append(actual)

        import sys
        sys.exit()

    evaluated_data = vectorised_data.mapValues(eval_func)

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
    parser.add_argument('--agg-func', default='max', const='all', nargs='?',
                        choices=['raw', 'max', 'or'],
                        help="The name of the SR aggregation method be one of the allowed methods")
    parser.add_argument('--algorithm', default='trueknowledge_sum', const='all', nargs='?',
                        choices=["truelearn_fixed", "truelearn_novel", "truelearn_interest", "truelearn_hybrid"],
                        help="The name of the algorithm can be one of the allowed algorithms")
    parser.add_argument("--num-topics", type=int, default=10,
                        help="The number of top ranked topics that have to be considered.")
    parser.add_argument('--skill-repr', default='cosine', const='all', nargs='?',
                        choices=['cosine', 'binary', "norm"],
                        help="How the skills should be represented in the bayesian_models")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Output directory path where the results will be saved.")
    parser.add_argument('--source-filepath', type=str, default=None,
                        help="where input data is")
    parser.add_argument('--engage-func', default='all', const='all', nargs='?',
                        choices=['all', 'sum', "quality"],
                        help="What engagement eval function to be used")
    parser.add_argument('--threshold', type=float, default=1.,
                        help="Probability threshold for classifying true")
    parser.add_argument('--def-var-factor', type=float, default=.5,
                        help="Default variance factor for knowledge")
    parser.add_argument('--i-def-var-factor', type=float, default=.5,
                        help="Default variance factor for interest")
    parser.add_argument('--var-constant', type=float, default=.5,
                        help="The constant value added to variance to avoid uncertainty shrinking")
    parser.add_argument('--tau-factor', type=float, default=.0,
                        help="Probability of watching even when cant learn")
    parser.add_argument('--interest-decay-type', default='short', const='all', nargs='?',
                        choices=['short', "long"],
                        help="Type of interest decay")
    parser.add_argument('--interest-decay-factor', type=float, default=.0,
                        help="Probability of watching even when cant learn")
    parser.add_argument('--beta-factor', type=float, default=.1,
                        help="Beta value for knowledge")
    parser.add_argument('--draw-probability', type=str, default="static",
                        help="Probability of drawing the match")
    parser.add_argument('--draw-factor', type=float, default=.1,
                        help="factor of draw probability to be used")
    parser.add_argument("--top-k-sr-topics", type=int, default=-1,
                        help="The number of top ranked topics that have to be considered for semantic relatedness.")
    parser.add_argument('--positive-only', action='store_true', help="learns from negative examples too")
    parser.add_argument('--prediction-only', action='store_true',
                        help="semantic relatedness is only considered at prediction time")
    parser.add_argument('--prob-combine-type', default='short', const='all', nargs='?',
                        choices=['and', "or", "weight", "acc_weight", "f1_weight", "meta-logistic", "meta-perceptron",
                                 "meta-truelearn", "meta-truelearn-greedy"],
                        help="Type of fucntion used to combine knowledge and interest")
    parser.add_argument('--know-prob', type=float, default=1.,
                        help="contribution from knowledge factor")
    parser.add_argument('--dilute-var', action='store_true', help="dilute variance")
    parser.add_argument('--dilution-factor', type=float, default=.0,
                        help="factor variance dilution to be enforced")
    parser.add_argument('--sr-func', default='raw', const='all', nargs='?', choices=['raw', 'pr', "gauss"],
                        help="What SR aggregation method is to be used")
    parser.add_argument('--is-video', action='store_true', help="if the prediction is done on full video or not")
    parser.add_argument('--n-jobs', type=str, default="*",
                        help="number of parallel jobs")
    parser.add_argument('--full-dataset', action='store_true', help="if full dataset or smaller dataset.")
    parser.add_argument('--debug', action='store_true', help="if debug mode?")
    parser.add_argument('--time', action='store_true', help="if timing mode")
    parser.add_argument('--topics', action='store_true', help="if related topics should be stored")
    parser.add_argument('--quality-mapping-filepath', type=str, required=True,
                        help="mapping of engagement values")
    parser.add_argument("--num-signals", type=int, default=0,
                        help="The number of events before which quality model is used.")
    parser.add_argument('--freq-type', default='k', const='all', nargs='?',
                        choices=['k', 'i', 'ki'],
                        help="The name of the algorithm can be one of the allowed algorithms")
    parser.add_argument('--freq-agg', default='sum', const='all', nargs='?',
                        choices=['sum', 'min', 'n_unique', 'n_events', 'n_vid'],
                        help="The name of the algorithm can be one of the allowed algorithms")
    args = vars(parser.parse_args())

    _ = main(args)
