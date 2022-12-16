import json
from os.path import join

import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

from analyses.truelearn_experiments.data_analysis.engagement_for_quality.analyse_engagement_of_learners import \
    get_event_n_statistics
from lib.spark_context import get_spark_context


def get_act_pred(line):
    act = int(line["actual"][0])
    pred = int(line["predicted"][0] >= .5)

    return (act, pred)


def get_1_stat(df):
    results = df.map(get_act_pred).collect()
    act, pred = zip(*results)

    acc = accuracy_score(act, pred)
    prec = precision_score(act, pred)
    rec = recall_score(act, pred)
    f1 = f1_score(act, pred)

    return acc, prec, rec, f1


def main(args):
    spark = get_spark_context(master="local[{}]".format(args["n_jobs"]))
    spark.sparkContext.setLogLevel("ERROR")

    with open(join(args["baseline_results_json_dir"], "model_results.json")) as infile:
        baseline_profiles = json.load(infile)

    with open(join(args["results_json_dir"], "model_results.json")) as infile:
        profiles = json.load(infile)

    # Load the dataset
    baseline_user_data = (spark.sparkContext.parallelize(baseline_profiles).repartition(30))
    user_data = (spark.sparkContext.parallelize(profiles).repartition(30))

    bl_acc, bl_prec, bl_rec, bl_f1 = get_1_stat(baseline_user_data)
    acc, prec, rec, f1 = get_1_stat(user_data)

    records = [
        {"Model": "TrueLearn Novel", "Acc.": bl_acc, "Prec.": bl_prec, "Rec.": bl_rec, "F1": bl_f1},
        {"Model": "TrueLearn Novel++", "Acc.": acc, "Prec.": prec, "Rec.": rec, "F1": f1}
    ]

    record_df = pd.DataFrame(records)
    record_df.to_csv(args["output_dir"])
    print()


if __name__ == '__main__':
    """this script takes in the wikified lectures file and the learner activity data from videolectures to build a .
    output of this script will be {slug, vid_id, part_id, start_time, stop_time, clean, text, wiki_concepts}
    eg: command to run this script:

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--baseline-results-json-dir', type=str, required=True,
                        help="the path the json file dir that contains results of the baseline model")
    parser.add_argument('--results-json-dir', type=str, required=True,
                        help="the path the json file dir that contains results")
    parser.add_argument('--n-jobs', type=str, default="*",
                        help="number of parallel jobs")
    # parser.add_argument('--test', default='vid-freq', const='all', nargs='?',
    #                     choices=['vid-freq', 'skill-freq', 'n-events'],
    #                     help="Type of fucntion used to combine knowledge and interest")

    args = vars(parser.parse_args())

    main(args)
