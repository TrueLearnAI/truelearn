import json
from os.path import join

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from analyses.quality_classifier.io_utils import GEN_FEATURES, COL_VER_3, LABEL_COLS, get_columned_dataset, DOMAIN_COL, \
    compute_domain
from lib.spark_context import get_spark_context
from scratch.generate_dataset.utils import MED_ENGAGEMENT_RATE


def prepare_vln_dataset(data):
    data[DOMAIN_COL] = data["categories"].apply(compute_domain)
    data["freshness"] = data["time"]
    data["silent_period_rate"] = data["fraction_silent_words"]
    data["speaker_speed"] = data["word_count"] / (data["total_lecture_duration"] / 60)
    data = data[data["language"] == "en"]

    return data


def get_training_lecture_ids(peek_test, vln):
    vln_ids = set(vln["slug"])
    test_peek_ids = set(peek_test["slug"])

    test_ids = vln_ids.intersection(test_peek_ids)
    train_ids = vln_ids.difference(test_ids)

    assert len(train_ids.intersection(test_ids)) == 0

    # remove sparse entries
    dense_vln_ids = set(vln[vln["num_learners"] >= 5]["slug"])
    final_train_ids = train_ids.intersection(dense_vln_ids)
    return final_train_ids, test_ids, train_ids


def train_quality_model(train_ids, test_ids, full_train_ids, vln, output_dir, n_jobs, is_test):
    columns = GEN_FEATURES + COL_VER_3
    columns += ["med_engagement"]

    vln = vln[columns]

    lectures, columns = get_columned_dataset(vln, col_cat=3)

    train_lectures = vln[vln["slug"].isin(train_ids)].dropna().reset_index()
    test_lectures = vln[vln["slug"].isin(test_ids)].dropna().reset_index()
    full_train_lectures = vln[vln["slug"].isin(full_train_ids)].dropna().reset_index()

    X_train, Y_train = train_lectures[columns], np.array(train_lectures["med_engagement"])
    X_test, Y_test = test_lectures[columns], np.array(test_lectures["med_engagement"])

    Y_train = np.log(Y_train)
    Y_test = np.log(Y_test)

    if not is_test:
        params = {'n_estimators': [100, 500, 750, 1000, 2000, 5000],
                  'max_depth': [3, 5, 10, 25]}
        folds = 5
        n_jobs = int(n_jobs) if n_jobs != "*" else 8

        print("no. of features: {}".format(X_train.shape[1]))
        print("training data size: {}".format(len(X_train)))
        print("testing data size: {}\n\n".format(len(X_test)))

        grid_model = GridSearchCV(RandomForestRegressor(), params, cv=folds, n_jobs=n_jobs, refit=True)
        grid_model.fit(X_train, Y_train)

        joblib.dump(grid_model, join(output_dir, "model.pkl"), compress=True)
    else:
        grid_model = joblib.load(join(output_dir, "model.pkl"))

    train_pred = grid_model.predict(X_train)
    test_pred = grid_model.predict(X_test)

    Y_train = np.exp(Y_train)
    train_pred = np.exp(train_pred)
    Y_test = np.exp(Y_test)
    test_pred = np.exp(test_pred)

    from sklearn import metrics as skm
    from scipy.stats import spearmanr

    train_rmse = np.sqrt(skm.mean_squared_error(Y_train, train_pred))
    test_rmse = np.sqrt(skm.mean_squared_error(Y_test, test_pred))

    train_spearman = spearmanr(Y_train, train_pred)
    test_spearman = spearmanr(Y_test, test_pred)

    print("Train RMSE: {} \n Train Spearman:{}".format(train_rmse, train_spearman))
    print("Test RMSE: {} \n Test Spearman:{}".format(test_rmse, test_spearman))

    # generate mappings
    test_mapping = test_lectures[["slug", "duration"]]
    test_mapping["prediction"] = test_pred
    test_mapping["actual"] = Y_test
    test_mapping["set"] = "test"

    # generate full train mapping
    X_train, Y_train = full_train_lectures[columns], np.array(full_train_lectures["med_engagement"])
    train_pred = np.exp(grid_model.predict(X_train))

    train_mapping = full_train_lectures[["slug", "duration"]]
    train_mapping["prediction"] = train_pred
    train_mapping["actual"] = Y_train
    train_mapping["set"] = "train"

    # merge mappings
    final_mapping = pd.concat([train_mapping, test_mapping], axis=0).reset_index()
    slug_to_stat_mapping = {
        "train": {}, "test": {}
    }
    for _, record in final_mapping.iterrows():
        slug = record["slug"]
        tmp_record = {
            "duration": record["duration"],
            "prediction": record["prediction"],
            "actual": record["actual"]
        }

        slug_to_stat_mapping[record["set"]][slug] = tmp_record

    return grid_model, slug_to_stat_mapping


def main(args):
    # load datasets
    vln = prepare_vln_dataset(pd.read_csv(args["vlengagement_dataset_path"]))

    peek_train = pd.read_csv(join(args["peek_dataset_path"], "session_data_validation.csv"),
                             names=["slug", "video_id", "part_id", "timestamp", "timeframe", "user",
                                    "t1", "s1", "t2", "s2", "t3", "s3", "t4", "s4", "t5", "s5", "label"])
    peek_test = pd.read_csv(join(args["peek_dataset_path"], "session_data_test.csv"),
                            names=["slug", "video_id", "part_id", "timestamp", "timeframe", "user",
                                   "t1", "s1", "t2", "s2", "t3", "s3", "t4", "s4", "t5", "s5", "label"])

    vln_train_slugs, vln_test_slugs, full_train_slugs = get_training_lecture_ids(peek_test, vln)

    quality_model, mapping = train_quality_model(vln_train_slugs, vln_test_slugs, full_train_slugs, vln,
                                                 args["output_dir"], args["n_jobs"], args["is_test"])

    with open(join(args["output_dir"], "engagement_mapping.json"), "w") as out:
        json.dump(mapping, out)


if __name__ == '__main__':
    """this script takes in the wikified lectures file and the learner activity data from videolectures to build a .
    output of this script will be {slug, vid_id, part_id, start_time, stop_time, clean, text, wiki_concepts}
    eg: command to run this script:

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--peek-dataset-path', type=str, required=True,
                        help="the path to where the PEEK dataset is")
    parser.add_argument('--vlengagement-dataset-path', type=str, required=True,
                        help="The path to VLN dataset with features")
    parser.add_argument('--slug-to-id-mapping-path', type=str, required=True,
                        help="The path to mapping that connects PEEK and VLN dataset")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="The path to models and results")
    parser.add_argument('--n-jobs', type=str, default="*",
                        help="number of parallel jobs")
    parser.add_argument('--is-test', action='store_true',
                        help="If only testing is done...")

    args = vars(parser.parse_args())

    main(args)
