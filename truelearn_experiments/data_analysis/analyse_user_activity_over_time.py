from datetime import datetime
from os.path import join

import numpy as np
import pandas as pd

from analyses.truelearn_experiments.utils import convert_to_records
from lib.spark_context import get_spark_context


def compute_gap(id, session):
    min_time = min([k["time"] for k in session])
    max_time = max([k["time"] for k in session])
    delta = datetime.fromtimestamp(max_time) - datetime.fromtimestamp(min_time)
    print("user: {}".format(id))
    print("num_events: {}".format(len(session)))
    print(delta.days)


def get_topic_mapping(filepath):
    data = pd.read_csv(filepath)
    mapping = {}
    for idx, record in data.iterrows():
        mapping[int(record['id'])] = record['title']

    return mapping


def generate_plot(id, session, topic_mapping, output_dir):
    data = []
    for s in session:
        data.append({
            "time": s["time"],
            "topic": s["topics"][0],
            "label": s["label"]
        })

    data_df = pd.DataFrame(data)
    data_df.sort_values(by=["time"], inplace=True, ignore_index=True)

    t_0 = min(data_df["time"])
    data_df["Time Difference"] = data_df["time"].apply(
        lambda t: datetime.fromtimestamp(t) - datetime.fromtimestamp(t_0))

    data_df["Topic Title"] = data_df["topic"].apply(lambda t: topic_mapping.get(t))
    data_df["Timestep"] = np.arange(len(data_df))

    one_hot = pd.get_dummies(data_df["Topic Title"]).T

    import seaborn as sns
    from matplotlib import pyplot as plt
    plt.clf()
    plt.figure(figsize=(35, 25))
    ax = sns.heatmap(one_hot, xticklabels=True, yticklabels=True,  label='small', cbar=False)
    plt.yticks(
        rotation=30,
        horizontalalignment='right',
        fontweight='light',
        fontsize='x-small')

    plt.xticks(
        rotation=45,
        horizontalalignment='right',
        fontweight='light',
        fontsize='x-small')

    plt.savefig(join(output_dir, "{}.svg".format(id)), format="svg")
    # plt.savefig(join(output_dir, "{}.png".format(id)), format="png")

def main(args):
    spark = get_spark_context(driver_memroy="30g", exectutor_memory="30g",
                              max_result_size="4g")
    spark.sparkContext.setLogLevel("ERROR")

    topic_mapping = get_topic_mapping(args["topic_mapping_filepath"])

    # load training data
    data = (spark.read.csv(args["dataset_filepath"], sep=",", header=False).
            rdd.
            # map(lambda l: convert_to_records(l, top_n=args["num_topics"], has_part_id=args["has_part_id"])))
            map(lambda l: convert_to_records(l, top_n=1, has_part_id=True)))

    # data2 = (spark.read.csv(join(args["dataset_filepath"], "session_data_validation.csv"), sep=",", header=False).
    #         rdd.
    #         map(lambda l: convert_to_records(l, top_n=args["num_topics"], has_part_id=True)))
    #
    # data = data1.union(data2)

    grouped_data = (data.map(lambda l: (l["session"], l)).
                    groupByKey(numPartitions=20).
                    mapValues(list).
                    filter(lambda l: len(l[1]) >= 5).
                    # filter(lambda l: any([i["label"] == 1 for i in l[1]])).
                    repartition(20))

    session_data = grouped_data.collectAsMap()

    session = session_data['16']
    # for id, session in session_data.items():
    generate_plot("16", session, topic_mapping, args["output_dir"])

    print()


if __name__ == '__main__':
    """this script takes in the wikified lectures file and the learner activity data from videolectures to build a .
    output of this script will be {slug, vid_id, part_id, start_time, stop_time, clean, text, wiki_concepts}
    eg: command to run this script:

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-filepath', type=str, required=True,
                        help="where training data is")
    parser.add_argument('--topic-mapping-filepath', type=str, required=True,
                        help="where training data is")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="where the output figures sit")
    args = vars(parser.parse_args())

    _ = main(args)
