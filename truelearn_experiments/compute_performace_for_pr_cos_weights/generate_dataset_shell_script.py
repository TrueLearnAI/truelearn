from os.path import join, sep

DEF_ENG_THRESH = 0.75
LOW_ENG_THRESH = 0.0
HIGH_ENG_THRESH = 1.0
MIN_EVENT_THRESH = 5

# _LECTURE_FOLDER_FORMAT = "atleast_5_sessions_cs_only_{}_lectures_with_cooccurence_all_topics.json"
# _SESSION_FOLDER_FORMAT = "timeframe_engagement_for_{}_lectures.json"

_OUTPUT_FILEPATH_FORMAT = "final_data_top_{}_topics_{}_sessions_{}_engagement_all_topics_pre_{}_pr_{}_cos_{}"
#
#
# _DATA_FILE_NAME = "session_data.csv"
# _CONFIG_FILE_NAME = "{}_{}.jsonl"  # algorithm and ranking

_GEN_DATASET_CMD_PATH = "scratch/generate_trueskill_data/"

# commands
MKDIR_CMD = "mkdir -p {}"
GEN_DATASET_CMD = "generate_final_datasets.py --wikified-data-filepath {}  --timerange-perc-engaged-filepath {} --output-dir {} --num-topics {} --top-n-sessions {} --engagement-threshold {} --lower-bound-perc-threshold {} --upper-bound-perc-threshold {} --min-events-threshold {} --topic-rank-func {} --pr-weight {} --cos-weight {}"
COMBOS = [
    (1, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "wx", 1.0, 0.0),
    (2, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "wx", 0.9, 0.1),
    (3, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "wx", 0.8, 0.2),
    (4, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "wx", 0.7, 0.3),
    (5, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "wx", 0.6, 0.4),
    (6, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "wx", 0.5, 0.5),
    (7, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "wx", 0.4, 0.6),
    (8, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "wx", 0.3, 0.7),
    (9, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "wx", 0.2, 0.8),
    (10, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "wx", 0.1, 0.9),
    (11, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "wx", 0.0, 1.0),
    (12, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "swx", 1.0, 0.0),
    (13, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "swx", 0.9, 0.1),
    (14, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "swx", 0.8, 0.2),
    (15, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "swx", 0.7, 0.3),
    (16, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "swx", 0.6, 0.4),
    (17, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "swx", 0.5, 0.5),
    (18, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "swx", 0.4, 0.6),
    (19, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "swx", 0.3, 0.7),
    (20, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "swx", 0.2, 0.8),
    (21, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "swx", 0.1, 0.9),
    (22, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "swx", 0.0, 1.0),
    (23, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "mwx", 1.0, 0.0),
    (24, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "mwx", 0.9, 0.1),
    (25, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "mwx", 0.8, 0.2),
    (26, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "mwx", 0.7, 0.3),
    (27, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "mwx", 0.6, 0.4),
    (28, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "mwx", 0.5, 0.5),
    (29, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "mwx", 0.4, 0.6),
    (30, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "mwx", 0.3, 0.7),
    (31, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "mwx", 0.2, 0.8),
    (32, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "mwx", 0.1, 0.9),
    (33, "part", 10, 20, DEF_ENG_THRESH, LOW_ENG_THRESH, HIGH_ENG_THRESH, MIN_EVENT_THRESH, "mwx", 0.0, 1.0)
]


def main(args):
    PYTHONPATH = args["python_path"] + " "

    commands = []

    # for each combination
    for idx, lecture_type, n_topics, n_sessions, eng_thresh, min_eng, max_eng, min_events, ranking, pr_w, cos_w in COMBOS:

        tmp_cmds = []
        tmp_cmds.append("echo generating_combo_{}".format(idx))

        # generate results filepaths
        if n_sessions == 20:
            N_SESSION_FOLDERPATH = "top_20_users"
        else:
            N_SESSION_FOLDERPATH = "full_dataset"

        LECTURE_TYPE_FOLDERPATH = lecture_type

        output_dataset_filename = _OUTPUT_FILEPATH_FORMAT.format(n_topics, n_sessions, int(eng_thresh * 100), ranking,
                                                                 pr_w, cos_w)

        results_dir_path = join(args["results_dir"], N_SESSION_FOLDERPATH, LECTURE_TYPE_FOLDERPATH,
                                output_dataset_filename)

        if not results_dir_path.endswith(sep):
            results_dir_path += sep

        tmp_cmds.append(MKDIR_CMD.format(results_dir_path))

        # generate paths
        input_lecture_dataset_path = join(args["lecture_data_dir"])
        input_session_dataset_path = join(args["session_data_dir"])

        # generate config
        gen_dataset_cmd = PYTHONPATH + join(_GEN_DATASET_CMD_PATH,
                                            GEN_DATASET_CMD.format(input_lecture_dataset_path,
                                                                   input_session_dataset_path, results_dir_path,
                                                                   n_topics, n_sessions, eng_thresh, min_eng, max_eng,
                                                                   min_events, ranking, pr_w, cos_w))
        tmp_cmds.append(gen_dataset_cmd)

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

    parser.add_argument('--lecture-data-dir', type=str, required=True,
                        help="where lecture data is")
    parser.add_argument('--session-data-dir', type=str, required=True,
                        help="where session data is")
    parser.add_argument('--results-dir', type=str, required=True,
                        help="where the results should be stored")
    parser.add_argument('--python-path', type=str, default="python",
                        help="where python env to run with")
    parser.add_argument('--shell-output-path', type=str, required=True,
                        help="where to output the shell script")

    args = vars(parser.parse_args())

    main(args)
