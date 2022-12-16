import json
from os.path import join


def make_concept_counts(line):
    from analyses.truelearn_experiments.utils import get_topic_dict

    _, records = line

    topics = {(topic, 1) for r in records for topic in get_topic_dict(r["topics"]).keys()}

    return list(topics)


def get_document_frequency_mapping(path):
    from lib.spark_context import get_spark_context
    from analyses.truelearn_experiments.run_experiments import MIN_NUM_EVENTS
    from analyses.truelearn_experiments.utils import convert_to_records

    # read the files
    spark = get_spark_context()
    spark.sparkContext.setLogLevel("ERROR")

    # load data
    data = (spark.read.csv(join(path, "session_data_validation.csv"), sep=",", header=False).
            rdd.
            map(lambda l: convert_to_records(l, top_n=5, has_part_id=True)))

    grouped_data = (data.map(lambda l: (l["session"], l)).
                    groupByKey(numPartitions=20).
                    mapValues(list).
                    filter(lambda l: len(l[1]) >= MIN_NUM_EVENTS))

    counts = grouped_data.flatMap(make_concept_counts).reduceByKey(lambda a, b: a + b).collectAsMap()
    N = grouped_data.count()

    counts["N"] = N

    return counts


def main(args):
    counts = get_document_frequency_mapping(args["dataset_filepath"])

    with open(join(args["output_dir"], "df_mapping.json"), "w") as out:
        json.dump(counts, out)


if __name__ == '__main__':
    """ This script finds the set of semantic relatedness issues

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-filepath', type=str, required=True,
                        help="where training data is")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Output directory path where the results will be saved.")
    parser.add_argument('--n-jobs', type=str, default="*", help="number of parallel jobs")

    args = vars(parser.parse_args())

    main(args)
