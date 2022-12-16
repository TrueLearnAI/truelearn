from os.path import join

import numpy as np
import pandas as pd


def main(args):
    users = pd.read_csv(join(args["dataset_filepath"], "session_id_mapping.csv"))
    users = users[users["role"] == "test"]
    users = users[users["num_events"] >= 100]
    users = users[users["num_events"] != 114]
    users["session"] = users["session_id"]
    users = users[["num_events", "perc_engaged", "session"]]

    ts_results = pd.read_csv(join(args["results_filepath"], "trueskill_single", "summary_results.csv"))
    ts_results_final = pd.DataFrame()

    for col in ts_results.columns:
        if col == "session" or col == "num_events":
            ts_results_final[col] = ts_results[col]
        else:
            ts_results_final["{}_{}".format(col, "ts")] = ts_results[col]

    tl_results = pd.read_csv(
        join(args["results_filepath"], "truelearn_novel_neg_orig", "_test", "userwise_metrics.csv"))

    tl_results_final = pd.DataFrame()
    for col in tl_results.columns:
        if col == "session" or col == "num_events":
            tl_results_final[col] = tl_results[col]
        else:
            tl_results_final["{}_{}".format(col, "tl")] = tl_results[col]

    result1 = pd.merge(users, ts_results_final, on="session")
    result1['num_events'] = result1['num_events_x']
    result_final = pd.merge(result1, tl_results_final, on="num_events")
    result_final = result_final[['perc_engaged', 'accuracy_ts', 'f1_ts',
                                 'precision_ts', 'recall_ts',
                                 'num_events', 'accuracy_tl', 'f1_tl', 'precision_tl', 'recall_tl',
                                 'topic_sparsity_rate_tl']]

    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    corr = result_final.corr("spearman")
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=1., annot=True, cbar_kws={"shrink": .5},
                annot_kws={"size": 10})

    print()


if __name__ == '__main__':
    """this script takes in the wikified lectures file and the learner activity data from videolectures to build a .
    output of this script will be {slug, vid_id, part_id, start_time, stop_time, clean, text, wiki_concepts}
    eg: command to run this script:

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-filepath', type=str, required=True,
                        help="where training data is")
    parser.add_argument('--results-filepath', type=str, required=True,
                        help="where training data is")
    # parser.add_argument('--algorithm', default='truelearn_novel', const='all', nargs='?',
    #                     choices=["truelearn_novel", "trueknowledge_sum",
    #                              "truelearn_background", "trueskill_single"],
    #                     help="The name of the algorithm can be one of the allowed algorithms")
    # parser.add_argument('--skill-repr', default='cosine', const='all', nargs='?',
    #                     choices=['cosine', 'binary', "norm"],
    #                     help="How the skills should be represented in the models")
    # parser.add_argument('--output-dir', type=str, required=True,
    #                     help="Output directory path where the results will be saved.")
    # parser.add_argument('--engage-func', default='all', const='all', nargs='?',
    #                     choices=['all', 'sum', "quality"],
    #                     help="What engagement eval function to be used")
    # parser.add_argument('--threshold', type=float, default=1.,
    #                     help="Probability threshold for classifying true")
    # parser.add_argument('--def-var-factor', type=float, default=.5,
    #                     help="Probability of knowing this topics")
    # parser.add_argument('--tau-factor', type=float, default=.0,
    #                     help="Probability of watching even when cant learn")
    # parser.add_argument('--interest-decay-factor', type=float, default=.0,
    #                     help="Probability of watching even when cant learn")
    # parser.add_argument('--beta-factor', type=float, default=.1,
    #                     help="Probability skipping even when can learn")
    # parser.add_argument('--draw-probability', type=str, default="static",
    #                     help="Probability of drawing the match")
    # parser.add_argument('--draw-factor', type=float, default=.1,
    #                     help="factor of draw probability to be used")
    # parser.add_argument("--num-topics", type=int, default=10,
    #                     help="The number of top ranked topics that have to be considered.")
    # parser.add_argument('--positive-only', action='store_true', help="learns from negative examples too")
    # parser.add_argument('--has-part-id', action='store_true', help="defines if the dataset has part ids")

    args = vars(parser.parse_args())

    _ = main(args)
