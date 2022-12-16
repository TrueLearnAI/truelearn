import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import ujson as json


def get_annotations(line, top_n):
    topics = [{
        "cosine": topic[0],
        "pagerank": topic[1],
        "title": topic[2]
    } for topic in line["annotations"][0]]

    topics.sort(key=lambda l: l["pagerank"], reverse=True)

    return topics[:top_n]


def filter_relevant_topics(topic, selected_topics):
    if selected_topics == 'all' or topic["title"] in selected_topics:
        yield topic


def main(args):
    topics = {str(i) for i in args["topics"].split("|")} if args["topics"] != "all" else "all"

    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # load training data
    data = (spark.sparkContext.textFile(args["dataset_filepath"], 300).
            map(json.loads).
            flatMap(lambda l: get_annotations(l, 10)).
            filter(lambda t: filter_relevant_topics(t, topics))).toDF()

    desc_data = (data.groupby("title").
                 agg(F.min("cosine"), F.max("cosine"), F.avg("cosine"), F.stddev_pop("cosine"), F.count("cosine")).
                 withColumn("perc_stdev", F.col("stddev_pop(cosine)") / F.col("avg(cosine)"))
                 ).toPandas()

    desc_data.sort_values(by=['count(cosine)'], inplace=True, ascending=False)
    desc_data.to_csv(args["output_dir"] + "lecture_cosine_distributions.csv", index=False)


if __name__ == '__main__':
    """Through this script, we want to check the relationship between topic coverage. We investigate the relationship 
    between engagement and normalised topic coverage for top 5 most prominent subjects in the 
    eg: command to run this script:

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-filepath', type=str, required=True,
                        help="where the lecture data is")
    parser.add_argument('--topics', type=str, default="all",
                        help="The number of top n topics used for analysing engagement")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Output file path where the results will be saved.")

    args = vars(parser.parse_args())

    main(args)
