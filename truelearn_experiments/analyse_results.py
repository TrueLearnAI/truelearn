import pandas as pd
from numpy import average

ADD_COLS = ['change_label_rate', 'num_topics', 'num_topics_rate', 'num_user_topics', 'positive_rate',
            'predict_positive_rate']


def main(args):
    results_dir = args["results_dir"]

    results = pd.DataFrame()
    summary = {}

    cols = set()

    algorithm = args["algorithm"]

    temp_df = pd.read_json(results_dir + algorithm + "_model_results.json").set_index("session")
    if args["weighted_avg"]:
        weights = temp_df["num_events"]
        summary_outfile = args["results_dir"] + "summary_accuracy_weighted.csv"
    else:
        weights = None
        summary_outfile = args["results_dir"] + "summary_accuracy.csv"

    temp_col = algorithm + "_accuracy"
    results[temp_col], summary[temp_col] = temp_df["accuracy"], average(temp_df["accuracy"], weights=weights)
    cols.add(temp_col)

    temp_col = algorithm + "_precision"
    results[temp_col], summary[temp_col] = temp_df["precision"], average(temp_df["precision"], weights=weights)
    cols.add(temp_col)

    temp_col = algorithm + "_recall"
    results[temp_col], summary[temp_col] = temp_df["recall"], average(temp_df["recall"], weights=weights)
    cols.add(temp_col)

    temp_col = algorithm + "_f1"
    results[temp_col], summary[temp_col] = temp_df["f1"], average(temp_df["f1"], weights=weights)
    cols.add(temp_col)

    if algorithm.startswith("trueknowledge") or algorithm.startswith("truelearn"):
        for col in ADD_COLS:
            results[col] = temp_df[col]

    cols = list(cols)
    cols.sort()

    results.to_csv(args["results_dir"] + "summary_results.csv")
    pd.DataFrame([summary], columns=cols).to_csv(summary_outfile, index=False)


if __name__ == '__main__':
    """this script takes in the wikified lectures file and the learner activity data from videolectures to build a .
    output of this script will be {slug, vid_id, part_id, start_time, stop_time, clean, text, wiki_concepts}
    eg: command to run this script:

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--results-dir', type=str, required=True,
                        help="where results data is")
    parser.add_argument('--algorithm', default='trueknowledge_all', const='all', nargs='?',
                        choices=["engage", "trueknowledge_all", "trueknowledge_sum", "truelearn_background",
                                 "persistent",
                                 "majority", "truelearn_bern_beta", "truelearn_novel", "truelearn_novel_interest"],
                        help="The name of the algorithm can be one of the allowed algorithms")
    parser.add_argument('--weighted-avg', action='store_true', help="The average used to report the accuracy values")

    args = vars(parser.parse_args())

    main(args)
