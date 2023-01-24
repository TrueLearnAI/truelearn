from collections import defaultdict
from os.path import join

import mpmath
import trueskill
import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from analyses.truelearn_experiments.utils import convert_to_records, get_topic_dict, get_videowise_data, \
    get_semantic_relatedness_mapping, get_related_skill_set, get_semantic_skill_inference, team_sum_quality, \
    STATIC_DRAW_PROB
from lib.spark_context import get_spark_context

import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

mpmath.mp.dps = 250  # sets higher precision in trueskill


def get_accuracy_values(user_actual, user_predicted, user_counts, user_unique_topic_count,
                        threshold=0.5):
    users = list(set(user_actual.keys()))
    users.sort()

    weights = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    metric_records = []

    for user in users:
        actual = user_actual[user]
        predicted = [int(i >= threshold) for i in user_predicted[user]]

        num_unique_topics = int(user_unique_topic_count[user])
        num_events = int(user_counts[user])

        weights.append(num_events)

        tmp_accuracy_score = accuracy_score(actual, predicted, normalize=True)
        tmp_precision_score = precision_score(actual, predicted)
        tmp_recall_score = recall_score(actual, predicted)
        tmp_f1_score = f1_score(actual, predicted)

        accuracies.append(tmp_accuracy_score)
        precisions.append(tmp_precision_score)
        recalls.append(tmp_recall_score)
        f1s.append(tmp_f1_score)

        metric_records.append({
            "session": user,
            "accuracy": tmp_accuracy_score,
            "precision": tmp_precision_score,
            "recall": tmp_recall_score,
            "f1": tmp_f1_score,
            "topic_sparsity_rate": num_unique_topics / float(num_events),
            "num_events": int(num_events)
        })

    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1s)

    metrics_df = pd.DataFrame([{
        "accuracy": avg_accuracy,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1
    }])

    avg_accuracy_w = np.average(accuracies, weights=weights)
    avg_precision_w = np.average(precisions, weights=weights)
    avg_recall_w = np.average(recalls, weights=weights)
    avg_f1_w = np.average(f1s, weights=weights)

    metrics_w_df = pd.DataFrame([{
        "accuracy_w": avg_accuracy_w,
        "precision_w": avg_precision_w,
        "recall_w": avg_recall_w,
        "f1_w": avg_f1_w
    }])

    metrics_df = metrics_df[["accuracy", "precision", "recall", "f1"]]
    metrics_w_df = metrics_w_df[["accuracy_w", "precision_w", "recall_w", "f1_w"]]
    return pd.DataFrame(metric_records), metrics_df, metrics_w_df


def get_sr_enabled_params(model, semantic_mapping, topic, def_mu, def_var, sr_aggr_func, top_k_sr_topics,
                          sr_func):
    updated_params = get_semantic_skill_inference(model, semantic_mapping, topic, def_mu,
                                                  def_var, sr_aggr_func, top_k_sr_topics, sr_func)

    return updated_params


def predict_and_model(user_event_count, user_unique_topic_count, user_models,
                      engageability_models, all_topics, events, is_single, def_mu, def_var, def_beta, is_video,
                      semantic_mapping, top_k_sr_topics, sr_func, sr_aggr_func, pred_only):
    tmp_user_predicted, tmp_user_actual = defaultdict(list), defaultdict(list)

    rel_topics = {}  # to track related topics

    # initiate trueskill
    trueskill.setup(mu=def_mu, sigma=np.sqrt(def_var), beta=def_beta, tau=0., draw_probability=STATIC_DRAW_PROB,
                    backend="mpmath")

    if is_video:
        start_event = 1
    else:
        start_event = 2

    for idx, event in enumerate(events):

        slug = event["slug"]
        vid_id = event["vid_id"]
        part = event["part"]
        user = event["session"]

        topics = get_topic_dict(event["topics"]).keys()

        all_topics |= set(topics)

        user_unique_topic_count[user] |= set(topics)

        user_model = user_models.get(user, {"mean": {},
                                            "variance": {}
                                            })

        slug_vid_id = (slug, vid_id, part)
        # slug_vid_id = (slug, vid_id)

        topic_model = engageability_models.get(slug_vid_id, {"mean": {},
                                                             "variance": {}
                                                             })

        # create_teams
        team_learner = tuple()
        learner_weights = tuple()
        orig_team_learner = tuple()
        team_mean_learner = list()

        team_learner_mean_vec = list()
        team_learner_var_vec = list()

        team_content = tuple()
        content_weights = tuple()
        orig_team_content = tuple()
        team_mean_content = list()

        team_content_mean_vec = list()
        team_content_var_vec = list()

        topic_seq = []

        if is_single:
            topics = [0]  # in single mode, there is only one skill. Hence, 1 topic (0)

        for topic in topics:
            topic_seq.append(topic)

            # if topic is not known by either learner nor the topic model
            if topic not in user_model["mean"] or topic not in topic_model["mean"]:

                if semantic_mapping is not None:
                    # it is fresh
                    _rel_topics = get_related_skill_set(user_model, semantic_mapping, topic, top_k_sr_topics,
                                                        sr_func=sr_func)
                else:
                    _rel_topics = {}

                _topic = topic
                rel_topics[topic] = {
                    "idx": idx,
                    "rel_topics": _rel_topics
                }

            # get user skill rating
            tmp_learner_skill = user_model["mean"].get(topic, def_mu)
            tmp_learner_sd = np.sqrt(user_model["variance"].get(topic, def_var))

            # for reinitialising learner if predict only
            orig_learner_skill = trueskill.Rating(mu=tmp_learner_skill, sigma=tmp_learner_sd)
            orig_team_learner += (orig_learner_skill,)

            if semantic_mapping is not None and tmp_learner_skill == def_mu and tmp_learner_sd == np.sqrt(def_var):
                updated_learner = get_sr_enabled_params(user_model, semantic_mapping, topic, def_mu, def_var,
                                                        sr_aggr_func, top_k_sr_topics, sr_func)

                tmp_learner_skill, tmp_learner_sd = updated_learner

            # used for prediction
            learner_skill = trueskill.Rating(mu=float(tmp_learner_skill), sigma=float(tmp_learner_sd))

            team_learner += (learner_skill,)
            learner_weights += (1.,)
            team_mean_learner.append(learner_skill.mu)

            team_learner_mean_vec.append(tmp_learner_skill)
            team_learner_var_vec.append(np.square(tmp_learner_sd))

            # get skill coverage
            tmp_coverage = topic_model["mean"].get(topic, def_mu)
            tmp_content_sd = np.sqrt(topic_model["variance"].get(topic, def_var))

            # for reinitialising learner if predict only
            orig_content_cov = trueskill.Rating(mu=tmp_coverage, sigma=tmp_content_sd)
            orig_team_content += (orig_content_cov,)

            if semantic_mapping is not None and tmp_coverage == def_mu and tmp_content_sd == np.sqrt(def_var):
                updated_content = get_sr_enabled_params(topic_model, semantic_mapping, topic, def_mu, def_var,
                                                        sr_aggr_func, top_k_sr_topics, sr_func)

                tmp_content_skill, tmp_content_sd = updated_content

            # used for prediction
            topic_cov = trueskill.Rating(mu=tmp_coverage, sigma=float(tmp_content_sd))

            team_content += (topic_cov,)
            content_weights += (1.,)
            team_mean_content.append(topic_cov.mu)

            team_content_mean_vec.append(tmp_coverage)
            team_content_var_vec.append(np.square(tmp_learner_sd))

        # check if user engages
        pred_prob = trueskill.quality([team_learner, team_content], weights=[learner_weights, content_weights])
        # engage if player is better than content, hence sum quality rather than draw...
        # pred_prob = team_sum_quality(np.array(team_learner_mean_vec),
        #                              np.array(team_learner_var_vec),
        #                              np.array(team_content_mean_vec),
        #                              np.array(team_content_var_vec), def_beta)
        # prediction = int(pred_prob >= .5)

        label = event["label"]

        if pred_only:
            # if prediction only, reload original params
            team_learner = orig_team_learner
            team_content = orig_team_content

        # # update
        # if label == 1:
        #     # learner wins
        #     new_team_learner, new_team_content = trueskill.rate([team_learner, team_content], ranks=[0, 1],
        #                                                         weights=[learner_weights, content_weights])
        # else:
        #     # content wins
        #     new_team_content, new_team_learner = trueskill.rate([team_content, team_learner], ranks=[0, 1],
        #                                                         weights=[content_weights, learner_weights])

        # update
        if label == 1:  # its a draw
            new_team_learner, new_team_content = trueskill.rate([team_learner, team_content], ranks=[0, 0],
                                                                weights=[learner_weights, content_weights])
        else:
            difference = np.sum(team_mean_learner) - np.sum(team_mean_content)

            if difference > 0.:  # learner wins --> boring content
                new_team_learner, _ = trueskill.rate([team_learner, team_content], ranks=[0, 1],
                                                     weights=[learner_weights, content_weights])
            elif difference < 0.:  # learner loses --> intimidation
                _, new_team_learner = trueskill.rate([team_content, team_learner], ranks=[0, 1],
                                                     weights=[content_weights, learner_weights])
            else:
                new_team_learner = team_learner

        # recreate user and content model representations
        for _idx, topic in enumerate(topic_seq):
            user_model["mean"][topic], user_model["variance"][
                topic] = new_team_learner[_idx].mu, new_team_learner[_idx].sigma ** 2

            topic_model["mean"][topic], topic_model["variance"][
                topic] = new_team_learner[_idx].mu, new_team_learner[_idx].sigma ** 2

        # assign it to the new dictionary
        user_models[user] = user_model
        engageability_models[slug_vid_id] = topic_model

        # if not first element, calculate accuracy
        user_event_count[user] += 1
        if user_event_count[user] >= start_event:
            tmp_user_actual[user].append(label)
            tmp_user_predicted[user].append(pred_prob)

    return user_event_count, user_unique_topic_count, tmp_user_predicted, tmp_user_actual, user_models, engageability_models, all_topics


def fit_model(spark, data_path, threshold=0.5, is_single=False, is_tracking=False, has_part_id=False, num_topics=10,
              is_video=False, sr_mapping=None, sr_topics_n=-1, sr_func=None, sr_aggr_func=None, pred_only=False):
    data = (spark.read.csv(data_path, sep=",", header=False).
            rdd.
            map(lambda l: convert_to_records(l, top_n=num_topics, has_part_id=has_part_id)))

    timewise_data = data.collect()

    timewise_data.sort(key=lambda l: (l["time"], l["timeframe"], l["part"]))

    def_mu = float(25.)
    def_sigma = float(def_mu / 3.)
    def_var = np.square(def_sigma)
    def_beta = float(def_sigma / 2.)

    # initiate performance recorders
    user_event_count = defaultdict(int)
    user_unique_topic_count = defaultdict(set)

    # initiate learner learning
    user_models = {}

    # initiate video, topic params
    engageability_models = {}

    all_topics = set()

    # if is_video:
    #     timewise_data = get_videowise_data(timewise_data, is_trueskill=True)
    # else:
    #     timewise_data = [timewise_data]

    # for bundled_event in timewise_data:
    (user_event_count, user_unique_topic_count, tmp_user_predicted, tmp_user_actual, user_models,
     engageability_models,
     all_topics
     ) = predict_and_model(user_event_count, user_unique_topic_count, user_models, engageability_models, all_topics,
                           timewise_data, is_single, def_mu, def_var, def_beta, is_video, sr_mapping, sr_topics_n,
                           sr_func, sr_aggr_func, pred_only)

    user_actual_results = tmp_user_actual
    user_predicted_results = tmp_user_predicted

    user_actual, user_predicted = defaultdict(list), defaultdict(list)

    # for idx, _ in enumerate(timewise_data):
    #     local_user_actual_results = user_actual_results[idx]
    #     local_user_predicted_results = user_predicted_results[idx]

    for user in user_actual_results.keys():
        user_actual[user] = user_actual_results[user]
        user_predicted[user] = user_predicted_results[user]

    user_unique_topic_count = {user_id: len(unique_topics) for user_id, unique_topics in
                               user_unique_topic_count.items()}

    if is_tracking:
        return user_actual, user_predicted, user_event_count

    accuracy_per_user, accuracy, weighted_accuracy = get_accuracy_values(user_actual, user_predicted, user_event_count,
                                                                         user_unique_topic_count, threshold=threshold)

    return accuracy_per_user, accuracy, weighted_accuracy


def main(args):
    spark = get_spark_context(driver_memroy="4g", exectutor_memory="4g", max_result_size="4g", lookup_timeout="300s")
    spark.sparkContext.setLogLevel("ERROR")

    if not args["semantic_relatedness_filepath"] is None:
        # load the semantic mapping
        semantic_mapping = get_semantic_relatedness_mapping(args["semantic_relatedness_filepath"])
    else:
        semantic_mapping = None

    # load testing data
    data_path = join(args["dataset_filepath"], "session_data_test.csv")

    if args["is_single"]:
        accuracy_per_user, accuracy, weighted_accuracy = fit_model(spark, data_path, is_single=True)
    else:
        accuracy_per_user, accuracy, weighted_accuracy = fit_model(spark, data_path, is_single=False,
                                                                   num_topics=args["num_topics"],
                                                                   sr_mapping=semantic_mapping, sr_topics_n=-1,
                                                                   sr_func=args["sr_func"], sr_aggr_func="or",
                                                                   pred_only=args["prediction_only"],
                                                                   has_part_id=args["has_part_id"])

        accuracy_per_user.to_csv(join(args["output_dir"], "summary_results.csv"), index=False)
        accuracy.to_csv(join(args["output_dir"], "summary_accuracy.csv"), index=False)
        weighted_accuracy.to_csv(join(args["output_dir"], "summary_accuracy_weighted.csv"), index=False)


if __name__ == '__main__':
    """this script takes in the wikified lectures file and the learner activity data from videolectures to build a .
    output of this script will be {slug, vid_id, part_id, start_time, stop_time, clean, text, wiki_concepts}
    eg: command to run this script:

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-filepath', type=str, required=True,
                        help="where training data is")
    # parser.add_argument('--sr-agg-func', default='max', const='all', nargs='?',
    #                     choices=['raw', 'max', 'or'],
    #                     help="The name of the SR aggregation method be one of the allowed methods")
    parser.add_argument('--semantic-relatedness-filepath', type=str, default=None,
                        help="where training data is")
    parser.add_argument("--num-topics", type=int, default=10,
                        help="The number of top ranked topics that have to be considered.")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Output directory path where the results will be saved.")
    parser.add_argument('--is-single', action='store_true')
    parser.add_argument('--has-part-id', action='store_true')
    parser.add_argument('--sr-func', default='raw', const='all', nargs='?', choices=['raw', 'pr', "gauss"],
                        help="What SR aggregation method is to be used")
    parser.add_argument('--prediction-only', action='store_true',
                        help="semantic relatedness is only considered at prediction time")
    parser.add_argument("--top-k-sr-topics", type=int, default=-1,
                        help="The number of top ranked topics that have to be considered for semantic relatedness.")

    args = vars(parser.parse_args())

    main(args)
