# pylint: disable=missing-function-docstring
# noqa
"""This module can be used to compare the running time and memory consumption \
of KnowledgeClassifier and NoveltyClassifier in the truelearn library and \
the original repository `https://github.com/sahanbull/TrueLearn`.
"""
import argparse
import numpy as np
import trueskill

from truelearn import learning, datasets


# fmt: off
def _erfc(x):
    """Complementary error function (via `http://bit.ly/zOLqbc`)."""
    z = abs(x)
    t = 1. / (1. + z / 2.)
    r = t * np.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (
            0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (
            0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (
            -0.82215223 + t * 0.17087277
    )))
    )))
    )))
    return 2. - r if x < 0 else r
# fmt: on


def _cdf(x, mu=0, sigma=1):
    """Cumulative distribution function."""
    return 0.5 * _erfc(-(x - mu) / (sigma * np.sqrt(2)))


def team_sum_quality(
    mean_skill_user, var_skill_user, mean_skill_content, var_skill_content, beta
):
    difference = np.sum(mean_skill_user) - np.sum(mean_skill_content)
    std = np.sqrt(np.sum(var_skill_user) + np.sum(var_skill_content) + beta)
    return _cdf(difference, 0, std)


# pylint: disable=too-many-arguments, too-many-locals
def truelearn_knowledge_original(
    records,
    positive_only=True,
    init_skill=0.0,
    def_var=0.5,
    tau=0.0,
    beta_sqr=0.35**2,
    threshold=0.5,
):
    trueskill.setup(
        mu=0.0,
        sigma=np.sqrt(1e-9),
        beta=float(np.sqrt(beta_sqr)),
        tau=tau,
        draw_probability=0.0,
        backend="mpmath",
    )

    user_model = {"mean": {}, "variance": {}}

    predicted = []

    for _, (topic_dict, label) in enumerate(records):
        # create_teams
        team_learner = tuple()
        team_learner_mean_vec = []
        team_learner_var_vec = []

        team_content = tuple()
        team_content_mean_vec = []
        team_content_var_vec = []

        topic_seq = []

        for topic, coverage in topic_dict.items():
            topic_seq.append(topic)
            # get user skill rating
            tmp_learner_mean = user_model["mean"].get(topic, init_skill)
            tmp_learner_var = user_model["variance"].get(topic, def_var)
            learner_skill = trueskill.Rating(
                mu=float(tmp_learner_mean), sigma=float(np.sqrt(tmp_learner_var))
            )

            team_learner += (learner_skill,)
            team_learner_mean_vec.append(tmp_learner_mean)
            team_learner_var_vec.append(tmp_learner_var)

            # get skill coverage
            tmp_coverage = coverage
            tmp_content_var = 1e-9
            topic_cov = trueskill.Rating(
                mu=tmp_coverage, sigma=float(np.sqrt(tmp_content_var))
            )
            team_content += (topic_cov,)
            team_content_mean_vec.append(tmp_coverage)
            team_content_var_vec.append(tmp_content_var)

        # check if user engages
        pred_prob = team_sum_quality(
            team_learner_mean_vec,
            team_learner_var_vec,
            team_content_mean_vec,
            team_content_var_vec,
            beta_sqr,
        )

        prediction = int(pred_prob >= threshold)

        # if label is negative and setting is positive only, skip updating
        if positive_only and label != 1:
            pass
        else:
            if label == 1:
                # learner wins
                new_team_learner, _ = trueskill.rate(
                    [team_learner, team_content], ranks=[0, 1]
                )
            else:
                # content wins
                _, new_team_learner = trueskill.rate(
                    [team_content, team_learner], ranks=[0, 1]
                )

            for _idx, topic in enumerate(topic_seq):
                user_model["mean"][topic], user_model["variance"][topic] = (
                    new_team_learner[_idx].mu,
                    new_team_learner[_idx].sigma ** 2,
                )

        predicted.append(prediction)


def truelearn_novelty_original(
    records,
    positive_only=False,
    init_skill=0.0,
    def_var=0.5,
    tau=0.0,
    beta_sqr=0.35**2,
    threshold=0.5,
    draw_probability="dynamic",
    draw_factor=0.1,
):
    user_model = {"mean": {}, "variance": {}}

    actual = []
    predicted = []

    for _, (topic_dict, label) in enumerate(records):
        # setup trueskill environment
        if draw_probability == "static":
            # _draw_probability = float(0.5932538086581619)  # population success rate
            _draw_probability = 1.0
        else:
            # compute novelty prob
            _draw_probability = float(np.mean(actual)) if actual else 0.0
            _draw_probability = (
                _draw_probability if _draw_probability > 0.0 else 1e-9
            )  # cant be zero
            _draw_probability = (
                _draw_probability if _draw_probability < 1.0 else 0.999999999
            )  # cant be one

        _draw_probability *= draw_factor

        trueskill.setup(
            mu=0.0,
            sigma=np.sqrt(def_var),
            beta=float(np.sqrt(beta_sqr)),
            tau=tau,
            draw_probability=_draw_probability,
            backend="mpmath",
        )

        # create_teams
        team_learner = tuple()
        team_mean_learner = []

        team_content = tuple()
        team_mean_content = []

        topic_seq = []

        for topic, coverage in topic_dict.items():
            topic_seq.append(topic)
            # get user skill rating
            tmp_learner_skill = user_model["mean"].get(topic, init_skill)
            learner_skill = trueskill.Rating(
                mu=tmp_learner_skill,
                sigma=np.sqrt(user_model["variance"].get(topic, def_var)),
            )

            team_learner += (learner_skill,)
            team_mean_learner.append(tmp_learner_skill)

            # get skill coverage
            tmp_content_topic = coverage
            topic_cov = trueskill.Rating(mu=tmp_content_topic, sigma=np.sqrt(1e-9))
            team_content += (topic_cov,)
            team_mean_content.append(tmp_content_topic)

        # check if user engages
        pred_prob = trueskill.quality([team_learner, team_content])
        prediction = int(pred_prob >= threshold)

        # if label is negative and setting is positive only, skip updating
        if positive_only and label != 1:
            pass
        else:
            # if positive
            if label == 1:
                new_team_learner, _ = trueskill.rate(
                    [team_learner, team_content], ranks=[0, 0]
                )
            else:  # if the person is not engaged...
                # check if the winner is learner or content
                difference = np.sum(team_mean_learner) - np.sum(team_mean_content)

                if difference > 0.0:  # learner wins
                    new_team_learner, _ = trueskill.rate(
                        [team_learner, team_content], ranks=[0, 1]
                    )
                elif difference < 0.0:  # learner loses
                    _, new_team_learner = trueskill.rate(
                        [team_content, team_learner], ranks=[0, 1]
                    )
                else:
                    new_team_learner = team_learner

            for _idx, topic in enumerate(topic_seq):
                user_model["mean"][topic], user_model["variance"][topic] = (
                    new_team_learner[_idx].mu,
                    new_team_learner[_idx].sigma ** 2,
                )

        predicted.append(prediction)
        actual.append(label)


def truelearn_knowledge_lib(event_label_pairs):
    knowledge_classifier = learning.KnowledgeClassifier()
    predictions = []

    _, event_label_pairs = event_label_pairs

    for event, label in event_label_pairs:
        predictions.append(knowledge_classifier.predict(event))
        knowledge_classifier.fit(event, label)


def truelearn_novelty_lib(event_label_pairs):
    novelty_classifier = learning.NoveltyClassifier()
    predictions = []

    _, event_label_pairs = event_label_pairs

    for event, label in event_label_pairs:
        predictions.append(novelty_classifier.predict(event))
        novelty_classifier.fit(event, label)


def driver(data_for_all, eval_func):
    for data_for_one in data_for_all:
        eval_func(data_for_one)


def main(args):
    _, test, _ = datasets.load_peek_dataset(train_limit=0)

    # prepare data for truelearn library
    top_n = args["num"]
    event_label_pairs_for_all = test
    event_label_pairs_for_all.sort(key=lambda t: len(t[1]), reverse=True)
    event_label_pairs_for_all = event_label_pairs_for_all[:top_n]

    # prepare data for original code
    records_for_all = [
        [
            (
                {
                    topic_id: knowledge.mean
                    for topic_id, knowledge in event.knowledge.topic_kc_pairs()
                },
                label,
            )
            for event, label in events
        ]
        for _, events in event_label_pairs_for_all
    ]

    if args["k"]:
        if args["original"]:
            driver(records_for_all, truelearn_knowledge_original)
        else:
            driver(event_label_pairs_for_all, truelearn_knowledge_lib)
    elif args["n"]:
        if args["original"]:
            driver(records_for_all, truelearn_novelty_original)
        else:
            driver(event_label_pairs_for_all, truelearn_novelty_lib)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", action="store_true", help="run knowledge classifier")
    parser.add_argument("-n", action="store_true", help="run novelty classifier")

    # if False, run truelearn library
    parser.add_argument(
        "--original", action="store_true", help="run original code", default=False
    )

    # select top-n most active learners
    parser.add_argument("--num", type=int, help="run benchmark on #n learners")

    main(vars(parser.parse_args()))
