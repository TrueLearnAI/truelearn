# DATA_DIR = "C:\Users\in4maniac\Documents\Data\x5gon\trueskill_data\final_datasets\top_20_users\full_lecture\"

# RESULTS_DIR = ""
from os.path import sep, join

_DATA_FOLDER_FORMAT = "final_data_top_{}_topics_{}_sessions_75_engagement_all_topics_pre_{}_pr_{}_cos_{}"
_DATA_FILE_NAME = "session_data.csv"
_CONFIG_FILE_NAME = "{}_{}.jsonl"  # algorithm and ranking

_GRID_SEARCH_PATH = "analyses/truelearn_experiments/grid_search/"

# commands
MKDIR_CMD = "mkdir -p {}"
GRID_CONFIG_CMD = "generate_grid_search_config.py --exp-dataset-filepath {} --exp-skill-repr {} {} --exp-algorithm {} --configs-file {}"
GRID_RUN_CMD = "run_grid_search.py --configs-file {} --results-dir {}"
GRID_SUMMARY_CMD = "get_result_summary.py --results-dir {}"

# truelearn_bernaulli experiments
truelearn_bernaulli_combos_pos = [
    (1, 5, 20, "wx", 0.4, 0.6, "truelearn_background", "binary", True),
    (2, 10, 20, "wx", 0.4, 0.6, "truelearn_background", "binary", True)
]

truelearn_bernaulli_combos_neg = [
    (1, 5, 20, "wx", 0.4, 0.6, "truelearn_background", "binary", False),
    (2, 10, 20, "wx", 0.4, 0.6, "truelearn_background", "binary", False)
]

truelearn_gaussian_combos_pos = [
    (1, 5, 20, "wx", 0.4, 0.6, "trueknowledge_sum", "cosine", True),
    (2, 10, 20, "wx", 0.4, 0.6, "trueknowledge_sum", "cosine", True)
]

truelearn_gaussian_combos_neg = [
    (1, 5, 20, "wx", 0.4, 0.6, "trueknowledge_sum", "cosine", False),
    (2, 10, 20, "wx", 0.4, 0.6, "trueknowledge_sum", "cosine", False)
]

truelearn_gaussian_all_combos = [
    (1, 5, 20, "wx", 0.4, 0.6, "trueknowledge_all", "cosine", True),
    (2, 10, 20, "wx", 0.4, 0.6, "trueknowledge_all", "cosine", True)
]

final_truelearn_bernaulli_combos = [
    (1, 5, "all", "wx", 0.4, 0.6, "truelearn_background", "binary", True),
    (2, 5, "all", "wx", 0.4, 0.6, "truelearn_background", "binary", False)
]

final_truelearn_gaussian_combos = [
    (1, 5, "all", "wx", 0.4, 0.6, "trueknowledge_sum", "cosine", True),
    (2, 5, "all", "wx", 0.4, 0.6, "trueknowledge_sum", "cosine", False)
]

final_truelearn_novel_combos = [
    (1, 5, "all", "wx", 0.4, 0.6, "truelearn_novel", "cosine", False),
]

final_truelearn_novel_interest_combos = [
    (1, 5, 20, "wx", 0.4, 0.6, "truelearn_novel_interest", "cosine", False),
]

pos_truelearn_novel_combos = [
    (1, 5, 20, "wx", 0.4, 0.6, "truelearn_novel", "cosine", False),
]

# # pr vs cos weight evaluation
pr_cos_weights_combos = [
    (1, 10, 20, "pr", 1.0, 0.0, "truelearn_background", "binary", True),
    (2, 10, 20, "wx", 0.8, 0.2, "truelearn_background", "binary", True),
    (3, 10, 20, "wx", 0.6, 0.4, "truelearn_background", "binary", True),
    (4, 10, 20, "wx", 0.4, 0.6, "truelearn_background", "binary", True),
    (5, 10, 20, "wx", 0.2, 0.8, "truelearn_background", "binary", True),
    (6, 10, 20, "cos", 0.0, 1.0, "truelearn_background", "binary", True),
]


def main(args):
    PYTHONPATH = args["python_path"] + " "

    commands = []

    _algorithm = args["algorithm"]

    if _algorithm == "truelearn_background_pos":
        combos = truelearn_bernaulli_combos_pos
    elif _algorithm == "trueknowledge_sum_pos":
        combos = truelearn_gaussian_combos_pos
    elif _algorithm == "truelearn_background_neg":
        combos = truelearn_bernaulli_combos_neg
    elif _algorithm == "trueknowledge_sum_neg":
        combos = truelearn_gaussian_combos_neg
    elif _algorithm == "trueknowledge_all":
        combos = truelearn_gaussian_all_combos
    elif _algorithm == "truelearn_background_final":
        combos = final_truelearn_bernaulli_combos
    elif _algorithm == "trueknowledge_sum_final":
        combos = final_truelearn_gaussian_combos
    elif _algorithm == "truelearn_novel_final":
        combos = final_truelearn_novel_combos
    elif _algorithm == "truelearn_novel_20":
        combos = pos_truelearn_novel_combos
    elif _algorithm == "truelearn_novel_interest":
        combos = final_truelearn_novel_interest_combos
    elif _algorithm == "pr_wx_cos":
        combos = pr_cos_weights_combos

    # for each combination
    for idx, num_topics, num_sessions, ranking_func, pr_w, cos_w, algorithm, skill_repr, pos_only in combos:

        tmp_cmds = []
        tmp_cmds.append("echo generating_combo_{}".format(idx))

        # "final_data_top_{}_topics_{}_sessions_75_engagement_all_topics_pre_{}_pr_{}_cos_{}"
        input_data_filename = _DATA_FOLDER_FORMAT.format(num_topics, num_sessions, ranking_func, pr_w, cos_w)
        if not input_data_filename.endswith(sep):
            input_data_filename += sep

        # generate results filepaths
        temp_algo_name = algorithm
        if not pos_only:
            temp_algo_name += "_neg"

        results_dir_path = join(args["results_dir"], input_data_filename, temp_algo_name)
        if not results_dir_path.endswith(sep):
            results_dir_path += sep

        tmp_cmds.append(MKDIR_CMD.format(results_dir_path))

        # generate paths
        input_dataset_path = join(args["dataset_filepath"], input_data_filename, _DATA_FILE_NAME)
        config_file_path = join(results_dir_path, _CONFIG_FILE_NAME.format(algorithm, ranking_func))

        if pos_only:
            pos_only = "--exp-positive-only"
        else:
            pos_only = ""

        # generate config
        grid_config_cmd = PYTHONPATH + join(_GRID_SEARCH_PATH,
                                            GRID_CONFIG_CMD.format(input_dataset_path, skill_repr, pos_only, algorithm,
                                                                   config_file_path))
        tmp_cmds.append(grid_config_cmd)

        # run grid search
        run_grid_cmd = PYTHONPATH + join(_GRID_SEARCH_PATH, GRID_RUN_CMD.format(config_file_path, results_dir_path))
        tmp_cmds.append(run_grid_cmd)

        # summarise result of grid search
        summarise_grid_cmd = PYTHONPATH + join(_GRID_SEARCH_PATH,
                                               GRID_SUMMARY_CMD.format(results_dir_path))
        tmp_cmds.append(summarise_grid_cmd)

        tmp_cmds.append("echo completed_combo_{}".format(idx))

        commands.append("\n\n".join(tmp_cmds))

    shell_script = "\n\n\n".join(commands)

    with open(args["shell_output_path"], "w") as outfile:
        outfile.write(shell_script)


if __name__ == '__main__':
    """this script takes in the wikified lectures file and the learner activity data from videolectures to build a .
    output of this script will be {slug, vid_id, part_id, start_time, stop_time, clean, text, wiki_concepts}
    eg: command to run this script:

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-filepath', type=str, required=True,
                        help="where training data is")
    parser.add_argument('--results-dir', type=str, required=True,
                        help="where the results should be stored")
    parser.add_argument('--python-path', type=str, default="python",
                        help="where python env to run with")
    parser.add_argument('--shell-output-path', type=str, required=True,
                        help="where to output the shell script")
    parser.add_argument('--algorithm', default='trueknowledge_sum_pos', const='all', nargs='?',
                        choices=['pr_wx_cos', "trueknowledge_sum_pos", "truelearn_novel_final", "truelearn_novel_20",
                                 "truelearn_background_pos", "trueknowledge_sum_neg", "truelearn_background_neg",
                                 "truelearn_background_final", "trueknowledge_sum_final", "truelearn_novel_interest"],
                        help="The name of the algorithm can be one of the allowed algorithms")

    args = vars(parser.parse_args())

    main(args)
