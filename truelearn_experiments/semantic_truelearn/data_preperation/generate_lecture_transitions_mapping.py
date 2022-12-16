import itertools
import os
from os.path import join

from analyses.truelearn_experiments.utils import convert_to_records
from lib.spark_context import get_spark_context


def reformat_file(line):
    _, records = line
    records.sort(key=lambda l: l["time"])

    records = {(record["slug"], record["vid_id"], record["part"]) for record in records}

    combos = itertools.combinations(records, 2)

    for item in records:
        yield (item, 1)

    for x, y in combos:
        yield ((x, y), 1)
        yield ((y, x), 1)  # to make it symmetric


def get_lecture_transition(dataset_filepath):
    from analyses.truelearn_experiments.run_experiments import MIN_NUM_EVENTS

    # read the files
    spark = get_spark_context()
    spark.sparkContext.setLogLevel("ERROR")

    # load data
    data = (spark.read.csv(join(dataset_filepath, "session_data_validation.csv"), sep=",", header=False).
            rdd.
            map(lambda l: convert_to_records(l, 1, has_part_id=True)))

    grouped_data = (data.map(lambda l: (l["session"], l)).
                    groupByKey(numPartitions=20).
                    mapValues(list).
                    filter(lambda l: len(l[1]) >= MIN_NUM_EVENTS))

    inverted_data = grouped_data.flatMap(reformat_file).reduceByKey(lambda a, b: a + b).collectAsMap()

    return inverted_data
