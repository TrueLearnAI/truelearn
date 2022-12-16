import os
from os.path import join, isdir
import pandas as pd


def main(args):
    runs = [join(args["results_dir"], o) for o in os.listdir(args["results_dir"])
            if isdir(join(args["results_dir"], o)) and o.startswith("final_data")]

    results = []

    for run in runs:
        params = run.split("_")
        pr_w, cos_w = params[-3], params[-1]
        grid_search_results_path = join(run, "truelearn_background", "grid_search_results.csv")
        tmp_result = pd.read_csv(grid_search_results_path)
        for _, record in tmp_result.iterrows():
            results.append({
                "pr_w": pr_w,
                "cos_w": cos_w,
                "beta_factor": record["beta_factor"],
                "tau_factor": record["tau_factor"],
                "accuracy": record["accuracy"],
                "precision": record["precision"],
                "recall": record["recall"],
                "f1": record["f1"],
            })

    results = pd.DataFrame(results)
    results = results[['pr_w', 'cos_w', 'beta_factor', 'tau_factor', 'accuracy', 'precision', 'recall', 'f1']]

    results.to_csv(join(args["results_dir"], "overall_results.csv"), index=False)


if __name__ == '__main__':
    """this script takes in the wikified lectures file and the learner activity data from videolectures to build a .
    output of this script will be {slug, vid_id, part_id, start_time, stop_time, clean, text, wiki_concepts}
    eg: command to run this script:

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--results-dir', type=str, required=True,
                        help="where results data is")
    args = vars(parser.parse_args())

    main(args)
