from functools import partial
from os.path import join

import pandas as pd
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from analyses.truelearn_experiments.run_experiments import vectorise_data
from analyses.truelearn_experiments.run_experiments_seq import fit_model
from analyses.truelearn_experiments.trueknowledge_recommender_models import truelearn_novel_model, \
    get_default_variance_from_coverage_values
from analyses.truelearn_experiments.utils import convert_to_records, get_topic_dict
from lib.spark_context import get_spark_context


def generate_eval_values(actual, predicted, window):
    precisions = []
    recalls = []
    f1s = []

    for i in range(len(predicted) - window + 1):
        tmp_actual = actual[i:i + window]
        tmp_predicted = predicted[i:i + window]

        precisions.append(precision_score(tmp_actual, tmp_predicted))
        recalls.append(recall_score(tmp_actual, tmp_predicted))
        f1s.append(f1_score(tmp_actual, tmp_predicted))

    return precisions, recalls, f1s


def eval_algorithm_truelearn(algorithm, session, window=10):
    time_cost, actual, predicted, num_records = algorithm(session)

    precs, recs, f1s = generate_eval_values(actual, predicted, window)

    return time_cost, precs, recs, f1s


def generate_metrics(record):
    (user, (_, actual, predicted, num_events)) = record
    init_f1 = f1_score(actual[:100], predicted[:100])
    fin_f1 = f1_score(actual[-100:], predicted[-100:])

    return {
        "user": user,
        "num_events": num_events,
        "init_f1": init_f1,
        "fin_f1": fin_f1
    }


def _eval_metric(act, pred, func, kwargs):
    tmp_metrics = []

    # for each bundle
    for idx in range(len(act)):
        tmp_act = act[idx]
        tmp_pred = pred[idx]
        tmp_metrics.append(func(tmp_act, tmp_pred, **kwargs))

    return np.mean(tmp_metrics)


def _generate_df_row(algorithm, num_events, metric, value):
    return {
        "Algorithm": algorithm,
        "num_events": num_events,
        "metric": metric,
        "value": value
    }


def generate_cumulative_metrics(user_actual, user_predicted, user_event_weights, max_events, algorithm, window=10):
    accs, precs, recs, f1s = [], [], [], []

    for num in range(1, max_events):
        num_events = num * window
        tmp_accs, tmp_precs, tmp_recs, tmp_f1s = [], [], [], []
        tmp_ws = []
        usr_cnt = 0
        for user, event_weight in user_event_weights.items():
            if len(user_actual[user]) < num_events:
                continue
            usr_cnt += 1
            tmp_actual = user_actual[user][:num_events]
            tmp_predicted = user_predicted[user][:num_events]

            if user == "2":
                print()

            tmp_acc = float(_eval_metric(tmp_actual, tmp_predicted, accuracy_score, {"normalize": True}))
            tmp_accs.append(tmp_acc)

            tmp_prec = float(_eval_metric(tmp_actual, tmp_predicted, precision_score, {}))
            tmp_precs.append(tmp_prec)

            tmp_rec = float(_eval_metric(tmp_actual, tmp_predicted, recall_score, {}))
            tmp_recs.append(tmp_rec)

            tmp_f1 = float(_eval_metric(tmp_actual, tmp_predicted, f1_score, {}))
            tmp_f1s.append(tmp_f1)

            tmp_ws.append(1)

        if usr_cnt == 0:
            break

        accs.append(_generate_df_row(algorithm, num_events, "Accuracy", np.average(tmp_accs, weights=tmp_ws)))
        precs.append(_generate_df_row(algorithm, num_events, "Precision", np.average(tmp_precs, weights=tmp_ws)))
        recs.append(_generate_df_row(algorithm, num_events, "Recall", np.average(tmp_recs, weights=tmp_ws)))
        f1s.append(_generate_df_row(algorithm, num_events, "F1", np.average(tmp_f1s, weights=tmp_ws)))

        if num_events >= max_events:
            break

    return accs + precs + recs + f1s


def compute_event_rates(actuals, max_events, algorithm, window=10):
    pos_rates = []

    for num in range(1, max_events):
        num_events = num * window

        tot_events = 0
        num_pos = []

        for user, actual in actuals.items():
            tmp_actual = actual[:num_events]

            pos_rate = np.mean(tmp_actual)
            num_pos.append(pos_rate)

            # tot_events += len(tmp_actual)
            # num_pos += sum(tmp_actual)

        pos_rates.append(
            {
                "Algorithm": algorithm,
                "num_events": num_events,
                "F1": np.mean(num_pos)
                # "F1": num_pos / tot_events
            })

        if num_events >= max_events:
            break

    return pos_rates


def compute_num_users(actuals, max_events, window=10):
    num_users = []

    for num in range(1, max_events):
        num_events = num * window
        usr_cnt = 0

        for user, actual in actuals.items():
            if len(actual) < num_events:
                continue

            usr_cnt += 1

        num_users.append(
            {
                "num_events": num_events,
                "Number of Learners": usr_cnt
            })

        if num_events >= max_events:
            break

    return num_users


def generate_unique_topic_counts(user_records, window=10, is_video=False):
    user, events = user_records

    max_events = len(events)

    if is_video:
        topic_vec_start = 6
    else:
        topic_vec_start = 1

    for num in range(1, max_events):
        num_events = num * window

        tmp_events = events[:num_events]

        topic_set = set()

        for event in tmp_events:
            # get the topics
            topic_vec = event[topic_vec_start:-1]
            topic_dict = get_topic_dict(topic_vec)
            topic_set |= set(topic_dict.keys())

        yield (num_events, [len(topic_set)])

        if num_events > max_events:
            break


def eval_algorithm_trueskill(spark, algorithm, sessions, session_id="0", window=10):
    user_actual, user_predicted = fit_model(spark, sessions, is_single=algorithm, is_tracking=True)

    actual = user_actual[session_id]
    predicted = user_predicted[session_id]

    precs, recs, f1s = generate_eval_values(actual, predicted, window)

    return precs, recs, f1s


def get_summary_dict(record):
    events, topics = record
    return {
        "Algorithm": "Avg. Unique Topics Per User",
        "num_events": events,
        "unique_topics": np.mean(topics)
    }


def main(args):
    spark = get_spark_context()
    spark.sparkContext.setLogLevel("ERROR")

    data = (spark.read.csv(join(args["dataset_filepath"], "session_data_test.csv"), sep=",", header=False).
            rdd.
            map(lambda l: convert_to_records(l, has_part_id=args["has_part_id"], top_n=args["num_topics"])))

    var_data = data.map(lambda l: (l["session"], l)).groupByKey(numPartitions=10).mapValues(list)

    def_var = float(get_default_variance_from_coverage_values(var_data, type="cosine"))

    if args["is_video"]:
        min_events = 2
        filter_func = lambda l: len(set([x[4] for x in l[1]])) >= min_events
    else:
        min_events = 50
        filter_func = lambda l: len(l[1]) >= min_events

    grouped_data = (data.
                    map(lambda l: (l["session"], l)).
                    groupByKey(numPartitions=10).
                    mapValues(list).
                    mapValues(lambda l: vectorise_data(l, args["skill_repr"], is_video=args["is_video"])).
                    filter(filter_func))

    # def_var = float(get_default_variance_from_coverage_values(0, type="binary"))
    #
    # grouped_data = (data.
    #                 map(lambda l: (l["session"], l)).
    #                 groupByKey(numPartitions=8).
    #                 mapValues(list).mapValues(lambda l: vectorise_data(l, "binary")))

    # get unique topic sets
    # a = grouped_data.filter(lambda l: len(l[1])> 280).first()
    # grouped_data = spark.sparkContext.parallelize([a])
    # f = list(generate_unique_topic_counts(a, window=10))
    # print()
    topic_histogram = (grouped_data.
                       flatMap(lambda l: generate_unique_topic_counts(l, window=10, is_video=args["is_video"])).
                       reduceByKey(lambda a, b: a + b).
                       map(get_summary_dict)).collect()

    # evaluate truelearn novel
    truelearn = partial(truelearn_novel_model, init_skill=0., def_var=float(def_var * 500.), tau=0., beta_sqr=0.5,
                        threshold=0.5, draw_probability="individual", positive_only=False,
                        draw_factor=0.01, tracking=True, is_video=args["is_video"])

    # results_tl = pd.DataFrame(grouped_data.mapValues(truelearn).map(generate_metrics).collect())
    results_tl = grouped_data.mapValues(truelearn).collect()

    user_actual_tl, user_predicted_tl = {}, {}
    users = set()
    for record in results_tl:
        (user, (_, actual, predicted, _)) = record
        user_actual_tl[user] = actual
        user_predicted_tl[user] = predicted
        users.add(user)

    # evaluate trueskill
    user_actual_ts, user_predicted_ts, user_event_count_ts = fit_model(spark,
                                                                       join(args["dataset_filepath"],
                                                                            "session_data_test.csv"),
                                                                       is_single=True, is_tracking=True,
                                                                       num_topics=args["num_topics"],
                                                                       has_part_id=args["has_part_id"],
                                                                       is_video=args["is_video"])

    user_actual_ts = {k: v for k, v in user_actual_ts.items() if k in users}
    user_predicted_ts = {k: v for k, v in user_predicted_ts.items() if k in users}
    if args["is_video"]:
        user_event_count_ts = {k: len(v) for k, v in user_actual_ts.items() if k in users}
    else:
        user_event_count_ts = {k: v for k, v in user_event_count_ts.items() if k in users}
    # user_actual_ts, user_predicted_ts, user_event_count_ts = {'3704': user_actual_ts['3704']}, {'3704': user_predicted_ts['3704']}, {'3704': user_event_count_ts['3704']}

    # calculate weights of users
    all_events = np.sum(list(user_event_count_ts.values()))
    max_events = np.max(list(user_event_count_ts.values()))

    user_event_weights = {k: v / all_events for k, v in user_event_count_ts.items()}

    # calculate the F1 scores
    values = []

    scores_ts = generate_cumulative_metrics(user_actual_ts, user_predicted_ts, user_event_weights, max_events,
                                            "Vanilla TrueSkill")

    values += scores_ts
    # user_event_weights = np.ones(len(user_actual_tl))
    scores_tl = generate_cumulative_metrics(user_actual_tl, user_predicted_tl, user_event_weights, max_events,
                                            "TrueLearn Novelty")

    values += scores_tl
    #
    # event_rates = compute_event_rates(user_actual_ts, max_events, "Positive Rate")
    #
    # values += event_rates
    #
    # pred_rates = compute_event_rates(user_predicted_ts, max_events, "Predictive Rate (TrueSkill)")
    #
    # values += pred_rates
    #
    # pred_rates = compute_event_rates(user_predicted_tl, max_events, "Predictive Rate (TrueLearn)")
    #
    # values += pred_rates

    results = pd.DataFrame(values)

    topic_df = pd.DataFrame(topic_histogram)

    num_users = compute_num_users(user_actual_ts, max_events)

    results2 = pd.DataFrame(num_users)
    results = results[results["metric"] == "F1"]

    # time, tl_precs, tl_recs, tl_f1 = eval_algorithm_truelearn(truelearn, active_session[1], window=WINDOW_SIZE)

    # # evaluate trueskill single
    # # algorithm = true for single trueskill
    # ts_precs, ts_recs, ts_f1 = eval_algorithm_trueskill(spark, True, args["dataset_filepath"], window=WINDOW_SIZE)

    # visualise
    # data = pd.DataFrame()
    #
    # data["Time Point"] = range(1, len(active_session[1]) - WINDOW_SIZE + 1)
    # data["Vanilla TrueSkill Precision"] = ts_precs
    # data["Vanilla TrueSkill Recall"] = ts_recs
    # data["Vanilla TrueSkill F1"] = ts_f1
    #
    # data["TrueLearn Novel Precision"] = tl_precs
    # data["TrueLearn Novel Recall"] = tl_recs
    # data["TrueLearn Novel F1"] = tl_f1
    #
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(style="ticks", palette="colorblind", context="paper")
    #
    ax1 = sns.lineplot(x="num_events", y="value", hue="Algorithm",
                       markers=True, dashes=False, data=results)

    ax = plt.twinx()

    # ax2 = sns.lineplot(x="num_events", y="Number of Learners", data=results2, ax=ax, color="black")
    ax2 = sns.lineplot(x="num_events", y="unique_topics", data=topic_df, ax=ax, color="black")
    print()
    # #
    # # ax2 = sns.lineplot(x="Time Point", y="TrueLearn Novel F1",
    # #                    markers=True, dashes=False, data=data)
    #
    # ax1 = sns.lineplot(x="Time Point", y="Vanilla TrueSkill F1",
    #                      data=data)
    #
    # ax2 = sns.lineplot(x="Time Point", y="TrueLearn Novel F1",
    #                       data=data)
    #
    # print()


if __name__ == '__main__':
    """this script takes in the wikified lectures file and the learner activity data from videolectures to build a .
    output of this script will be {slug, vid_id, part_id, start_time, stop_time, clean, text, wiki_concepts}
    eg: command to run this script:

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-filepath', type=str, required=True,
                        help="where training data is")
    parser.add_argument('--algorithm', default='truelearn_novel', const='all', nargs='?',
                        choices=["truelearn_novel", "trueknowledge_sum",
                                 "truelearn_background", "trueskill_single"],
                        help="The name of the algorithm can be one of the allowed algorithms")
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
    parser.add_argument("--num-topics", type=int, default=10,
                        help="The number of top ranked topics that have to be considered.")
    parser.add_argument('--positive-only', action='store_true', help="learns from negative examples too")
    parser.add_argument('--has-part-id', action='store_true', help="defines if the dataset has part ids")
    parser.add_argument('--is-video', action='store_true', help="if the clarifications are for videos")

    args = vars(parser.parse_args())

    _ = main(args)
