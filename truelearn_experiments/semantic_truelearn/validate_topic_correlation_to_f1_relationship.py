import pandas as pd
import ujson as json
from analyses.truelearn_experiments.run_experiments import restructure_data
import os
from functools import partial
import numpy as np

from analyses.truelearn_experiments.trueknowledge_recommender_models import get_default_variance_from_coverage_values, \
    truelearn_novel_model
from analyses.truelearn_experiments.utils import convert_to_records, vectorise_data
from lib.spark_context import get_spark_context


def get_truelearn_func(grouped_data):
    def_var_factor = 500.
    beta_factor = .5

    cosine_var = float(get_default_variance_from_coverage_values(grouped_data, "cosine"))
    _def_var = float(np.square(cosine_var * def_var_factor))
    _beta_sqr = float(np.square(np.sqrt(_def_var) * beta_factor))
    truelearn_func = partial(truelearn_novel_model, init_skill=0., def_var=_def_var, tau=.0, beta_sqr=_beta_sqr,
                             threshold=.5, draw_probability="individual", positive_only=False, draw_factor=.01)

    return truelearn_func


def get_performance_values(restructured_data):
    new_recs = []
    for i in restructured_data:
        new_recs.append({
            "id": i["session"],
            "f1": i["f1"]
        })
    return pd.DataFrame(new_recs)


def get_records(line):
    id = line["session"]
    corr = np.mean([i["relatedness"] for i in line["pairs"]])
    return (id, corr)


def main(args):
    spark = get_spark_context()
    spark.sparkContext.setLogLevel("ERROR")

    # get the correlation values
    corr = (spark.sparkContext.textFile(args["topic_relatedness_filepath"]).
            map(json.loads).
            map(get_records).
            groupByKey().
            map(lambda l: {"id": l[0], "relatedness": np.mean(list(l[1]))}))

    corr_df = pd.DataFrame(corr.collect())

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
                    mapValues(list))

    vectorised_data = grouped_data.mapValues(lambda l: vectorise_data(l, "cosine"))

    truelearn_func = get_truelearn_func(grouped_data)

    evaluated_data = vectorised_data.mapValues(truelearn_func)
    restructured_data = evaluated_data.map(restructure_data).collect()

    performance_vals = get_performance_values(restructured_data)

    final_df = performance_vals.merge(corr_df)
    print()

    # filter available relatedness values
    # tagme_mapping = pd.read_csv(args["topic_relatedness_filepath"])
    #
    # id_to_relatedness_mapping = {}
    # for _, record in tagme_mapping.iterrows():
    #     id_to_relatedness_mapping[record["id"]] = record["tagme_id"]
    #
    # id_to_tagme_mapping_b = spark.sparkContext.broadcast(id_to_tagme_mapping)
    #
    # modified_pairs = (grouped_data.
    #                   flatMap(lambda l: add_tagme_mappings(l, id_to_tagme_mapping_b)).
    #                   map(json.dumps).
    #                   repartition(5000))
    #
    # modified_pairs.saveAsTextFile(args["output_dir"], compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")


if __name__ == '__main__':
    """ This script finds the set of semantic relatedness issues

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-filepath', type=str, required=True,
                        help="where training data is")
    parser.add_argument('--topic-relatedness-filepath', type=str, required=True,
                        help="where training topic id mapping is")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Output directory path where the results will be saved.")

    args = vars(parser.parse_args())

    main(args)
