import os
import pandas as pd
import ujson as json

from analyses.truelearn_experiments.utils import convert_to_records, vectorise_data, get_topic_dict
from lib.spark_context import get_spark_context


def get_topic_pairs(line):
    """ get the topic pair tuples based on new topics and all the previous topics that are likely to be semantically
    related to them.

    Args:
        line (str, [{str: val}]): all the events per each learner

    Returns:
        [(str, str)]: list of tuples of (prev topic, current topic)

    """
    _, events = line

    # sort the events
    events.sort(key=lambda l: (l["time"], l["part"]))

    learned_topics = set()

    for event in events:
        topics = list(get_topic_dict(event["topics"], "cosine").keys())
        _tmp_event_learned_topics = set()

        for current_topic in topics:
            if current_topic not in learned_topics:
                for learned_topic in learned_topics:
                    yield (learned_topic, current_topic)
                _tmp_event_learned_topics.add(current_topic)

        learned_topics |= _tmp_event_learned_topics


def add_tagme_mappings(line, id_to_tagme):
    record = {"id_pair": line}
    a, b = line

    if a in id_to_tagme.value and b in id_to_tagme.value:
        record["tagme_pair"] = (id_to_tagme.value[a], id_to_tagme.value[b])
        yield record


def main(args):
    spark = get_spark_context()
    spark.sparkContext.setLogLevel("ERROR")

    # load data
    validation = spark.read.csv(os.path.join(args["dataset_filepath"], "session_data_validation.csv"), sep=",",
                                header=False)
    test = spark.read.csv(os.path.join(args["dataset_filepath"], "session_data_test.csv"), sep=",", header=False)

    data = validation.union(test)
    data = (data.
            rdd.
            map(lambda l: convert_to_records(l, top_n=5, has_part_id=True)))

    grouped_data = (data.
                    map(lambda l: (l["session"], l)).
                    groupByKey(numPartitions=20).
                    mapValues(list).
                    flatMap(get_topic_pairs).distinct())

    # filter available relatedness values
    tagme_mapping = pd.read_csv(args["topic_mapping_filepath"])

    id_to_tagme_mapping = {}
    for _, record in tagme_mapping.iterrows():
        id_to_tagme_mapping[record["id"]] = record["tagme_id"]

    id_to_tagme_mapping_b = spark.sparkContext.broadcast(id_to_tagme_mapping)

    modified_pairs = (grouped_data.
                      flatMap(lambda l: add_tagme_mappings(l, id_to_tagme_mapping_b)).
                      map(json.dumps).
                      repartition(5000))

    modified_pairs.saveAsTextFile(args["output_dir"], compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")


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

    args = vars(parser.parse_args())

    main(args)
