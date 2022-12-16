from functools import partial
from os.path import join
import numpy as np
import pandas as pd

from analyses.truelearn_experiments.trueknowledge_recommender_models import get_semantic_relatedness_mapping, \
    get_default_variance_from_coverage_values, truelearn_novel_model
from analyses.truelearn_experiments.utils import convert_to_records, vectorise_data, get_summary_stats
from lib.spark_context import get_spark_context


def _update_summary_stats_records(id, temp_actual, temp_predicted, idx, algorithm, stats):
    accuracy, precision, recall, f1, _ = get_summary_stats(temp_actual, temp_predicted, 0)

    stats.append({
        "Learner": int(id),
        "Algorithm": algorithm,
        "Number of Events": idx,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    return stats


def get_partitioned_metrics(records, num_events, algorithm, stats):
    for record in records:
        id, events = record
        _, actual, predicted, _ = events
        num_records = len(actual)
        temp_actual, temp_predicted = [], []
        for idx in range(num_records):
            temp_actual.append(actual[idx])
            temp_predicted.append(predicted[idx])

            if len(temp_actual) == num_events:
                stats = _update_summary_stats_records(id, temp_actual, temp_predicted, idx + 1, algorithm, stats)
                temp_actual, temp_predicted = [], []

        # remaining events
        stats = _update_summary_stats_records(id, temp_actual, temp_predicted, num_records, algorithm, stats)

    return stats


def generate_performance_figure(results_df, metric):
    # algorithms = results_df["algorithm"].unique()

    import seaborn as sns
    sns.set()
    palette = sns.color_palette("husl", 20)
    import matplotlib.pyplot as plt

    ax = sns.lineplot(x="Number of Events", y=metric, hue="Learner", style="Algorithm", data=results_df,
                      palette=palette)

    # for algorithm in algorithms:
    #    results = results_df[results_df["algorithm"] == algorithm]["learner", "event_num", metric]

    plt.show()


def main(args):
    spark = get_spark_context(master="local[{}]".format(args["n_jobs"]))
    spark.sparkContext.setLogLevel("ERROR")
    result_stats = []

    # load training data

    # data_val = (spark.read.csv(join(args["dataset_filepath"], "session_data_validation.csv"), sep=",", header=False).
    #             rdd)
    # data_test = (spark.read.csv(join(args["dataset_filepath"], "session_data_test.csv"), sep=",", header=False).
    #              rdd)

    # data = (data_val.union(data_test).
    #         map(lambda l: convert_to_records(l, top_n=1, has_part_id=True)))

    data = (spark.read.csv(join(args["dataset_filepath"], "session_data_test.csv"), sep=",", header=False).
            rdd.
            map(lambda l: convert_to_records(l, top_n=1, has_part_id=True)))

    grouped_data = data.map(lambda l: (l["session"], l)).groupByKey(numPartitions=10).mapValues(list).cache()

    cosine_var = float(get_default_variance_from_coverage_values(grouped_data, "cosine"))

    vectorised_data = grouped_data.mapValues(lambda l: vectorise_data(l, "cosine"))

    # truelearn novel
    _def_var = float(np.square(cosine_var * 10000.))
    _beta_sqr = float(np.square(np.sqrt(_def_var) * .5))
    _tau = float(1. * 0.)

    eval_func_basic = partial(truelearn_novel_model, init_skill=0., def_var=_def_var, tau=_tau, beta_sqr=_beta_sqr,
                              threshold=.5, draw_probability="individual", positive_only=False, draw_factor=0.25,
                              tracking=True)

    truelearn_results = vectorised_data.mapValues(eval_func_basic).collect()
    result_stats = get_partitioned_metrics(truelearn_results, 100, "TrueLearn", result_stats)

    # semantic truelearn
    semantic_mapping = get_semantic_relatedness_mapping(args["semantic_relatedness_filepath"])
    semantic_mapping_b = spark.sparkContext.broadcast(semantic_mapping)

    _def_var = float(np.square(cosine_var * 5000.))
    _beta_sqr = float(np.square(np.sqrt(_def_var) * .5))
    _tau = float(1. * 0.)

    eval_func_semantic = partial(truelearn_novel_model, init_skill=0., def_var=_def_var, tau=_tau, beta_sqr=_beta_sqr,
                                 threshold=.5, draw_probability="individual", positive_only=False, draw_factor=0.,
                                 semantic_mapping=semantic_mapping_b, agg_func="or", is_pred_only=True, is_diluted=True,
                                 dil_factor=0.01, tracking=True)

    semantic_truelearn_results = vectorised_data.mapValues(eval_func_semantic).collect()
    result_stats = get_partitioned_metrics(semantic_truelearn_results, 100, "Semantic TrueLearn", result_stats)

    results_df = pd.DataFrame(result_stats)

    generate_performance_figure(results_df, "Accuracy")
    generate_performance_figure(results_df, "Precision")
    generate_performance_figure(results_df, "Recall")
    generate_performance_figure(results_df, "F1 Score")

    print()

    # restructured_data = evaluated_data.map(restructure_data).collect()
    #
    # with open(join(args["output_dir"], "model_results.json"), "w") as outfile:
    #     json.dump(restructured_data, outfile)

    # return restructured_data


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
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Output directory path where the results will be saved.")
    parser.add_argument('--n-jobs', type=str, default="*",
                        help="number of parallel jobs")

    args = vars(parser.parse_args())

    _ = main(args)
