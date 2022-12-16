from os.path import join
import ujson as json
import pandas as pd

from analyses.truelearn_experiments.utils import convert_to_records, get_topic_dict
from lib.spark_context import get_spark_context


def load_kc_mapping(path):
    mapping_df = pd.read_csv(path)

    mapping = {}
    for _, record in mapping_df.iterrows():
        mapping[record["id"]] = record["tagme_id"]

    return mapping


def get_tagme_topics(topics, mapping_b):
    topics = set(get_topic_dict(topics).keys())
    mod_topics = [mapping_b.value.get(topic, None) for topic in topics]
    mod_topics = {topic for topic in mod_topics if topic is not None}

    return mod_topics


def main(args):
    spark = get_spark_context(master="local[{}]".format(args["n_jobs"]))
    spark.sparkContext.setLogLevel("ERROR")

    # load kc mapping
    mapping = load_kc_mapping(args["kc_to_tagme_mapping_filepath"])
    mapping_b = spark.sparkContext.broadcast(mapping)

    # load training data
    data = (spark.read.csv([join(args["dataset_filepath"], "session_data_validation.csv"),
                            join(args["dataset_filepath"], "session_data_test.csv")], sep=",", header=False).
            rdd.
            map(lambda l: convert_to_records(l, top_n=args["num_topics"], has_part_id=True)).
            map(lambda l: {"session": l["session"], "topics": get_tagme_topics(l["topics"], mapping_b)}))

    grouped_data = (data.
                    map(lambda l: (l["session"], l["topics"])).
                    reduceByKey(lambda a, b: a | b, numPartitions=20).
                    map(lambda l: {"session": l[0], "tagme_pair": list(l[1])}))

    grouped_data.map(json.dumps).saveAsTextFile(args["output_filepath"],
                                                compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")


if __name__ == '__main__':
    """this script takes in the wikified lectures file and the learner activity data from videolectures to build a .
    output of this script will be {slug, vid_id, part_id, start_time, stop_time, clean, text, wiki_concepts}
    eg: command to run this script:

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-filepath', type=str, required=True,
                        help="where training data is")
    parser.add_argument('--kc-to-tagme-mapping-filepath', type=str, required=True,
                        help="where the tagme mapping is")
    parser.add_argument("--num-topics", type=int, default=10,
                        help="The number of top ranked topics that have to be considered.")
    parser.add_argument('--n-jobs', type=str, default="*", help="number of parallel jobs")
    parser.add_argument('--output-filepath', type=str, required=True,
                        help="where the topic selection is stored in")

    args = vars(parser.parse_args())

    _ = main(args)
