import os

import pandas as pd
import ujson as json

from analyses.truelearn_experiments.utils import convert_to_records
from lib.api import get_tagme_title_resolution_response, get_tagme_semantic_relatedness_response
from lib.spark_context import get_spark_context


def get_wiki_topics(topic_vect, id_to_topic_mapping_b):
    n_topics = int(len(topic_vect) / 2)
    topics = []

    for i in range(n_topics):
        topic_idx = i * 2
        topics += [id_to_topic_mapping_b.value[topic_vect[topic_idx]]]

    return topics


def get_title_ids(title_topics, token):
    ids = []
    for title in title_topics:
        try:
            resp = get_tagme_title_resolution_response(title, token)
            ids.append(resp["wiki_id"])
        except ValueError:
            continue

    return ids


def get_correlations(t, id_to_topic_mapping_b, api_key):
    topics = get_wiki_topics(t["topics"], id_to_topic_mapping_b)

    # get tagme ids
    tagme_ids = get_title_ids(topics, api_key)

    # get correaltions
    correlations = get_tagme_semantic_relatedness_response(tagme_ids, "mw", api_key)
    t["pairs"] = correlations["pairs"]

    return t


def main(args):
    spark = get_spark_context(master="local[{}]".format(args["n_jobs"]))
    spark.sparkContext.setLogLevel("ERROR")

    tagme_token = args["tagme_token"]

    # filter available relatedness values
    topic_mapping = pd.read_csv(args["topic_mapping_filepath"])

    id_to_topic_mapping = {}
    for _, record in topic_mapping.iterrows():
        id_to_topic_mapping[record["id"]] = record["title"].replace(" ", "_")

    id_to_topic_mapping_b = spark.sparkContext.broadcast(id_to_topic_mapping)

    # load data
    validation = spark.read.csv(os.path.join(args["dataset_filepath"], "session_data_validation.csv"), sep=",",
                                header=False)
    test = spark.read.csv(os.path.join(args["dataset_filepath"], "session_data_test.csv"), sep=",", header=False)

    data = validation.union(test)
    data = (data.
            rdd.
            map(lambda l: convert_to_records(l, top_n=5, has_part_id=True))).repartition(100)

    enriched_data = (data.map(lambda l: get_correlations(l, id_to_topic_mapping_b, tagme_token)).
                     map(json.dumps))

    enriched_data.saveAsTextFile(args["output_dir"],
                                 compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")


if __name__ == '__main__':
    """ This script finds the set of semantic relatedness issues

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-filepath', type=str, required=True,
                        help="where training data is")
    parser.add_argument('--topic-mapping-filepath', type=str, required=True,
                        help="where training topic id mapping is")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Output directory path where the results will be saved.")
    parser.add_argument('--tagme-token', type=str, required=True,
                        help="Output directory path where the results will be saved.")
    parser.add_argument('--n-jobs', type=str, default="*", help="number of parallel jobs")

    args = vars(parser.parse_args())

    main(args)
