import itertools
import json
from os.path import join
import pandas as pd
import numpy as np
import networkx as nx

import os
import seaborn as sns
import matplotlib.pyplot as plt
from networkx import average_node_connectivity, minimum_node_cut, number_connected_components, bridges

from analyses.truelearn_experiments.grid_search.eval.summarise_experiment_results import decode_algo_name
from analyses.truelearn_experiments.trueknowledge_recommender_models import get_semantic_relatedness_mapping
from analyses.truelearn_experiments.utils import build_graph
from lib.spark_context import get_spark_context

truelearn_algos = ["truelearn_novel", "trueknowledge_sum_pos", "truelearn_background_pos"]
trueskill_algos = ["trueskill_single", "trueskill_multi"]

CUTOFF = 0


def get_topics(user_model, mapping):
    topics = {int(k) for k in user_model.keys()}
    combos = itertools.combinations(topics, 2)

    sr_vals = {}
    for src, dst in combos:
        sr = mapping.value.get((src, dst))
        if sr is not None:
            sr_vals[(src, dst)] = sr

    return topics, sr_vals


def get_graph_connectedness_values(topics, edges):
    nodes = list(topics)
    edges = [(k[0], k[1], v) for k, v in edges.items()]

    g = build_graph(nodes, edges)

    avg_connectivity = average_node_connectivity(g)
    is_connected = nx.is_connected(g)
    if is_connected:
        min_cut_set_size = len(minimum_node_cut(g))
    else:
        min_cut_set_size = None
    num_conn_comps = number_connected_components(g)
    num_bridges = len(list(bridges(g)))

    return avg_connectivity, min_cut_set_size, num_conn_comps, num_bridges


def get_user_records(user, mapping):
    # remove irrelevant fields
    _ = user.pop("roc_score")
    _ = user.pop("num_user_topics")

    # get topics
    user_model = user.pop("user_model")
    topics, relatedness = get_topics(user_model, mapping)

    # get avg. sr value
    avg_sr_value = np.mean(list(relatedness.values()))

    # get avg. connectedness
    avg_connectivity, min_cut_set_size, num_conn_comps, num_bridges = get_graph_connectedness_values(topics,
                                                                                                     relatedness)

    user["avg_sr_value"] = avg_sr_value
    user["connectedness"] = avg_connectivity
    user["min_cut_set_size"] = min_cut_set_size
    user["num_conn_comps"] = num_conn_comps
    user["num_bridges"] = num_bridges

    return user


def main(args):
    spark = get_spark_context(master="local[{}]".format(args["n_jobs"]))
    spark.sparkContext.setLogLevel("ERROR")

    semantic_mapping = get_semantic_relatedness_mapping(args["semantic_relatedness_filepath"])
    semantic_mapping_b = spark.sparkContext.broadcast(semantic_mapping)

    runs = [root for root, dirs, files in os.walk(args["results_dir"]) if "summary_metrics.csv" in files]

    for run in runs:
        with open(join(run, "model_results.json")) as infile:
            profiles = json.load(infile)

        user_data = (spark.sparkContext.parallelize(profiles).
                     repartition(1000))

        temp_user_models = user_data.map(lambda u: get_user_records(u, semantic_mapping_b)).collect()

        with open(join(run, "model_graph_results.json"), "w") as outfile:
            json.dump(temp_user_models, outfile)


if __name__ == '__main__':
    """this script takes in the wikified lectures file and the learner activity data from videolectures to build a .
    output of this script will be {slug, vid_id, part_id, start_time, stop_time, clean, text, wiki_concepts}
    eg: command to run this script:

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--results-dir', type=str, required=True,
                        help="the path to configs file where each line is a config dict")
    parser.add_argument('--semantic-relatedness-filepath', type=str, required=True,
                        help="where training data is")
    parser.add_argument('--n-jobs', type=str, default="*",
                        help="number of parallel jobs")
    args = vars(parser.parse_args())

    main(args)
