import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from analyses.truelearn_experiments.utils import convert_to_records


def main(args):
    topics = {str(i) for i in args["topics"].split("|")}

    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # load training data
    data = (spark.read.csv(args["dataset_filepath"], sep=",", header=False).
            rdd.
            map(convert_to_records).
            map(lambda l: l["topics"]))

    id_2_title_mapping = pd.read_csv(args["topic_id_to_text_mapping_filepath"])
    id_2_title_mapping = {float(rec["id"]): rec["title"] for _, rec in id_2_title_mapping.iterrows() if
                          rec["title"] in topics}

    # get desc stats
    num_topics = int(len(data.first()) / 2)

    desc_data = (data.
                 flatMap(
        lambda l: [(id_2_title_mapping[l[i * 2]], l[i * 2 + 1]) for i in range(num_topics) if
                   l[i * 2] in id_2_title_mapping]).
                 toDF().
                 withColumnRenamed("_1", "topic").
                 withColumnRenamed("_2", "cosine").
                 groupby("topic").
                 agg(F.min("cosine"), F.max("cosine"), F.avg("cosine"), F.stddev_pop("cosine"), F.count("cosine")).
                 withColumn("perc_stdev", F.col("stddev_pop(cosine)") / F.col("avg(cosine)"))
                 )

    desc_data.to_csv()
    print()


if __name__ == '__main__':
    """Through this script, we want to check the relationship between topic coverage. We investigate the relationship 
    between engagement and normalised topic coverage for top 5 most prominent subjects in the 
    eg: command to run this script:

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-filepath', type=str, required=True,
                        help="where the data is")
    parser.add_argument('--topic-id-to-text-mapping-filepath', type=str, required=True,
                        help="where the topic mapping is")
    parser.add_argument('--topics', type=str, required=True,
                        help="The number of top n topics used for analysing engagement")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Output file path where the results will be saved.")

    args = vars(parser.parse_args())

    main(args)
