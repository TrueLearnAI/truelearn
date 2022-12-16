from analyses.truelearn_experiments.semantic_truelearn.data_preperation.get_all_topics_for_user import load_kc_mapping
from lib.spark_context import get_spark_context

import ujson as json


def add_wiki_ids(line, mapping_b):
    pair_0 = mapping_b.value.get(line["tagme_pair"][0])
    pair_1 = mapping_b.value.get(line["tagme_pair"][1])

    if pair_0 is not None and pair_1 is not None:
        line["id_pair"] = [pair_0, pair_1]
        yield line


def main(args):
    mapping = load_kc_mapping(args["tag_me_to_id_mapping"])
    tagme_to_id_mapping = {}
    for wiki, tagme in mapping.items():
        tagme_to_id_mapping[tagme] = wiki

    spark = get_spark_context()
    spark.sparkContext.setLogLevel("ERROR")

    tagme_to_id_mapping_b = spark.sparkContext.broadcast(tagme_to_id_mapping)

    data = (spark.sparkContext.textFile(args["tag_me_pair_mapping"]).
            map(json.loads).
            flatMap(lambda l: add_wiki_ids(l, tagme_to_id_mapping_b)).
            map(json.dumps))

    data.saveAsTextFile(args["output_filepath"], compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")


if __name__ == '__main__':
    """this script takes in the wikified lectures file and the learner activity data from videolectures to build a .
    output of this script will be {slug, vid_id, part_id, start_time, stop_time, clean, text, wiki_concepts}
    eg: command to run this script:

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--tag-me-pair-mapping', type=str, required=True,
                        help="where the tagme pair mapping is")
    parser.add_argument('--tag-me-to-id-mapping', type=str, required=True,
                        help="where the csv of tagme id to normal id mapping is")
    parser.add_argument('--output-filepath', type=str, required=True,
                        help="where the topic selection is stored in")

    args = vars(parser.parse_args())

    _ = main(args)
