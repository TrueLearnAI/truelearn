from collections import defaultdict
from os.path import join

import ujson as json
import pandas as pd
from wordcloud import WordCloud

from analyses.truelearn_experiments.utils import get_semantic_relatedness_mapping
from lib.spark_context import get_spark_context


def save_wordcloud_figure(path, rel_topics):
    tmp_wc = WordCloud(background_color="white", max_words=50).generate_from_frequencies(rel_topics)
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(tmp_wc, interpolation='bilinear')
    plt.axis("off")

    plt.savefig(path, format="svg")


def main(args):
    spark = get_spark_context(master="local[{}]".format(8))
    spark.sparkContext.setLogLevel("ERROR")

    with open(args["dataset_filepath"]) as infile:
        learner_records = json.load(infile)

    semantic_mapping_top_50 = (spark.
                               sparkContext.
                               parallelize(
        get_semantic_relatedness_mapping(args["semantic_relatedness_filepath"]).items()).
                               map(lambda l: (str(l[0][0]), [(str(l[0][1]), l[1])])).
                               reduceByKey(lambda a, b: a + b).
                               mapValues(lambda l: sorted(l, key=lambda k: -k[1])[:50]).
                               mapValues(lambda l: {k[0]: k[1] for k in l})).collectAsMap()

    learner_records = spark.sparkContext.parallelize(learner_records)

    topic_learn_freq = (learner_records.
                        flatMap(lambda l: [(top, [rels]) for top, rels in l["rel_topics"].items() if len(rels) > 0]).
                        reduceByKey(lambda a, b: a + b, numPartitions=100).
                        collectAsMap()
                        )

    mapping = pd.read_csv(args["topic_id_to_text_mapping_filepath"])
    id_to_human_mappping = {}

    for _, record in mapping.iterrows():
        id_to_human_mappping[str(record["id"])] = record["title"]

    topic_stats = {}
    for topic, events in topic_learn_freq.items():

        most_related_50_topics = semantic_mapping_top_50.get(topic)
        if most_related_50_topics is None:
            continue

        num_learners = len(events)
        if num_learners < 50:
            continue

        weights = defaultdict(list)
        for event in events:
            for rel_topic, sr in event.items():
                if sr > 0:
                    weights[rel_topic].append(sr)

        topic_stats[topic] = {
            "num_learners": num_learners,
            "topic_weights": dict(weights),
            "top_50_weights": most_related_50_topics
        }

    ordered_topics = [{
        "topic_id": topic,
        "topic": id_to_human_mappping[topic],
        "freq": record["num_learners"]} for topic, record in topic_stats.items() if topic in id_to_human_mappping]
    ordered_df = pd.DataFrame(ordered_topics).sort_values(by=["freq"], ascending=False)
    ordered_df.reset_index(inplace=True, drop=True)

    ordered_df.to_csv(join(args["output_dir"], "_most_frequent_topics.csv"), index=False)

    for topic, record in topic_stats.items():
        # humanise things...
        human_topic = id_to_human_mappping.get(topic)
        if human_topic is None:
            continue
        top_50 = {id_to_human_mappping[rel_topic]: float(sr) for rel_topic, sr in record["top_50_weights"].items() if
                  rel_topic in id_to_human_mappping}
        topic_weights = {id_to_human_mappping[rel_topic]: float(sum(vals)) for rel_topic, vals in
                         record["topic_weights"].items() if rel_topic in id_to_human_mappping}

        if len(top_50) == 0 or len(topic_weights) == 0:
            continue

        # save figs
        save_wordcloud_figure(join(args["output_dir"], "{}_1_{}.svg".format(topic, human_topic.replace(" ", "_"))),
                              top_50)

        save_wordcloud_figure(join(args["output_dir"], "{}_2_{}.svg".format(topic, human_topic.replace(" ", "_"))),
                              topic_weights)


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
    parser.add_argument('--semantic-relatedness-filepath', type=str, required=True,
                        help="where training data is")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Output file path where the results will be saved.")

    args = vars(parser.parse_args())

    main(args)
