from collections import Counter
from os.path import join

import numpy as np
import pandas as pd

from analyses.truelearn_experiments.data_analysis.get_related_topic_distribution import save_wordcloud_figure


def get_activity_stats(data, label, records):
    learner_num_events = dict(Counter(list(data["user_id"])))

    for id, count in learner_num_events.items():
        if count >= 5:
            records.append({
                "user": id,
                "num_events": count,
                "Group": label
            })

    return records


def generate_plot_dataframe(records_df):
    num_events = list(records_df["num_events_bucket"].unique())

    records = []
    for bucket in num_events:
        num_users = records_df[records_df["num_events_bucket"] == bucket]
        num_train_users = len(num_users[num_users["Group"] == "Training"])
        num_test_users = len(num_users[num_users["Group"] == "Test"])

        records.extend([{
            "Number of Events": int(bucket),
            "Group": "Training Data",
            "Number of Learners": num_train_users
        }, {
            "Number of Events": int(bucket),
            "Group": "Test Data",
            "Number of Learners": num_test_users
        }])

    return pd.DataFrame(records)


def generate_num_event_plot(validation, test, output):
    records = []
    records = get_activity_stats(validation, "Training", records)
    records = get_activity_stats(test, "Test", records)

    records_df = pd.DataFrame(records)
    records_df["num_events_bucket"] = records_df["num_events"].apply(lambda x: np.ceil(x / 20) * 20)

    plot_df = generate_plot_dataframe(records_df)

    import seaborn as sns
    from matplotlib import pyplot as plt

    g = sns.barplot(x="Number of Events", y="Number of Learners",
                    hue="Group", data=plot_df)
    plt.xticks(rotation=45, ha='right')

    plt.savefig(join(output, "event_count_plot.svg"), format="svg")


def get_topic_mapping(path):
    mapping = pd.read_csv(path)

    final_mapping = {}
    for _, record in mapping.iterrows():
        final_mapping[int(record["id"])] = str(record["title"])

    return final_mapping


def generate_word_count(valid, test, id2topic_mapping, output_path):
    valid_list = []

    for _, record in valid.iterrows():
        key = int(record["t1"])
        if key in id2topic_mapping:
            valid_list.append(id2topic_mapping[key])

    test_list = []

    for _, record in test.iterrows():
        key = int(record["t1"])
        if key in id2topic_mapping:
            test_list.append(id2topic_mapping[key])

    dataset_list = valid_list + test_list

    save_wordcloud_figure(join(output_path, "topic_wordcloud_train.svg"), dict(Counter(valid_list)))
    save_wordcloud_figure(join(output_path, "topic_wordcloud_test.svg"), dict(Counter(test_list)))
    save_wordcloud_figure(join(output_path, "topic_wordcloud_dataset.svg"), dict(Counter(dataset_list)))


def get_lecture_stats(valid, test):
    valid_items = valid[["slug", "video_id", "part_id"]]
    test_items = test[["slug", "video_id", "part_id"]]

    valid_set = {(record["slug"], record["video_id"]) for _, record in valid_items.iterrows()}
    test_set = {(record["slug"], record["video_id"]) for _, record in test_items.iterrows()}

    print("Number of Unique Videos: {}".format(valid_set.union(test_set)))

    valid_set = {(record["slug"], record["video_id"], record["part_id"]) for _, record in valid_items.iterrows()}
    test_set = {(record["slug"], record["video_id"], record["part_id"]) for _, record in test_items.iterrows()}

    print("Number of Unique Video Fragments: {}".format(valid_set.union(test_set)))


def main(args):
    valid_datapath = join(args["dataset_dir"], "session_data_validation.csv")
    valid_data = pd.read_csv(valid_datapath, names=["slug", "video_id", "part_id", "time", "timeframe_id", "user_id",
                                                    "t1", "c1", "t2", "c2", "t3", "c3", "t4", "c4", "t5", "c5",
                                                    "label"])

    test_datapath = join(args["dataset_dir"], "session_data_test.csv")
    test_data = pd.read_csv(test_datapath, names=["slug", "video_id", "part_id", "time", "timeframe_id", "user_id",
                                                  "t1", "c1", "t2", "c2", "t3", "c3", "t4", "c4", "t5", "c5", "label"])

    if args["report"] == "events":
        generate_num_event_plot(valid_data, test_data, args["output_dir"])

    elif args["report"] == "wordcloud":
        id_to_topic_mapping = get_topic_mapping(join(args["dataset_dir"], "title_id_mapping.csv"))

        generate_word_count(valid_data, test_data, id_to_topic_mapping, args["output_dir"])
    else:
        get_lecture_stats(valid_data, test_data)


if __name__ == '__main__':
    """Through this script, we want to check the relationship between topic coverage. We investigate the relationship 
    between engagement and normalised topic coverage for top 5 most prominent subjects in the 
    eg: command to run this script:

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-dir', type=str, required=True,
                        help="where the dataset is")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Output file path where the results will be saved.")
    parser.add_argument('--report', default='max', const='all', nargs='?',
                        choices=['events', 'wordcloud', 'lectures'],
                        help="The report we are interested in")

    args = vars(parser.parse_args())

    main(args)
