import random
from collections import defaultdict
from datetime import datetime, timedelta

import mpmath
import numpy as np
import trueskill

from analyses.truelearn_experiments.utils import get_summary_stats, get_topic_dict, get_related_skill_set, \
    get_semantic_skill_inference, team_sum_quality, erfc, STATIC_DRAW_PROB, decode_vector, apply_interest_decay

# 22732: "RL",
# 5427: "CV",
# 17915: "NLP",
# 11323: "Global Warming"
# 18157: Neuroscience

# EXPLAINABLE_TOPIC_SET = frozenset([22732, 17915, 5427, 11323, 18157])
EXPLAINABLE_TOPIC_SET = frozenset([22732, 11323])

mpmath.mp.dps = 250  # sets higher precision in trueskill


def v_func(x):
    return (pdf(x) / cdf(x))


def w_func(x):
    v = v_func(x)
    return (v * (v + x))


def pdf(x, mu=0, sigma=1):
    """Probability density function"""
    return (1 / np.sqrt(2 * np.pi) * abs(sigma) *
            np.exp(-(((x - mu) / abs(sigma)) ** 2 / 2)))


def cdf(x, mu=0, sigma=1):
    """Cumulative distribution function"""
    return 0.5 * erfc(-(x - mu) / (sigma * np.sqrt(2)))


def dilute_variance(learner_mean, learner_std, user_model, semantic_mapping, topic, dil_factor, top_k_sr_topics,
                    def_var):
    related_skills = get_related_skill_set(user_model, semantic_mapping, topic, top_k_sr_topics)

    learner_var = learner_std ** 2

    if learner_var == float('Inf'):
        print()

    for skill, rel in related_skills.items():
        # learner_var *= (1. + (dil_factor * rel))
        for _ in range(int(np.ceil(np.log10(user_model["updates"][skill])))):
            # for _ in range(user_model["updates"][skill]):
            learner_var *= (1. + (dil_factor * rel))

    # make sure learner var is bigger than init var
    learner_var = max(learner_var, def_var)

    return learner_mean, np.sqrt(learner_var)


def get_default_variance_from_coverage_values(data, type="cosine"):
    # get variance of all topic coverage values in the dataset
    if type == "binary":
        return 1.0

    topic_coverage = np.var(data.
                            values().
                            flatMap(lambda events: [coverage for event in events
                                                    for coverage in
                                                    get_topic_dict(event["topics"], type=type).values()]).
                            collect())

    return topic_coverage


def compute_c(beta_sqr, skill_var):
    return np.sqrt(beta_sqr + skill_var)


def get_compatibility(skill_means_sum, skill_vars_sum, coverages_sum, beta_sqr):
    """uses trueskill compatibility equation to compute the compatibility between learner and content. Will add all the
    Gaussians as a team and compute compatibility

    Args:
        skill_means_sum:
        skill_vars_sum:
        coverages_sum:
        beta_sqr:

    Returns:
        float: prob compatibility
    """
    c = compute_c(beta_sqr, skill_vars_sum)
    d = beta_sqr / np.square(c)
    return np.sqrt(d) * np.exp(-np.square(skill_means_sum - coverages_sum) / (2 * np.square(c)))


def get_user_engageability(user, topics, method="all", threshold=1.0, def_user_var=10., beta_sqr=0.):
    """get a boolean value if the user is going to be engaged or not

    Args:
        user:
        topics:
        method:
        threshold:

    Returns:

    """
    if method == "all":
        num_topics = float(len(topics))
        positives = sum([user["mean"].get(topic, .0) >= cov for topic, cov in topics.items()])
        return bool(positives / num_topics >= threshold)

    elif method == "sum":
        return bool(sum([user["mean"].get(topic, .0) - cov for topic, cov in topics.items()]) >= .0)

    elif method == "quality":

        competence = list()

        # calculate the vectors required
        for topic, cov in topics.items():
            user_mean = user["mean"].get(topic, .0)
            user_var = user["variance"].get(topic, def_user_var)

            topic_mean = cov

            compatibility = get_compatibility(user_mean, user_var, topic_mean, beta_sqr)
            competence.append(compatibility >= 0.5)

        overall_competence = sum(competence) / len(competence)
        return bool(overall_competence >= threshold)


def convert_mappings_to_vectors(user_model, topic_dict, def_var):
    # create the vectors for the new model
    topic_seq = []
    u_mean = []
    u_var = []
    t_cov = []

    # dict to array
    for topic, coverage in topic_dict.items():
        topic_seq.append(topic)
        u_mean.append(user_model["mean"].get(topic, 0))
        u_var.append(user_model["variance"].get(topic, def_var))
        t_cov.append(coverage)

    return topic_seq, np.array(u_mean), np.array(u_var), np.array(t_cov)


def update_truelearn_fixed(label, team_learner, team_content):
    if label == 1:
        # learner wins
        new_team_learner, _ = trueskill.rate([team_learner, team_content], ranks=[0, 1])
    else:
        # content wins
        _, new_team_learner = trueskill.rate([team_content, team_learner], ranks=[0, 1])

    return new_team_learner


def update_truelearn_novel(label, team_learner, team_content, learner_weights, content_weights, team_mean_learner,
                           team_mean_content):
    if label == 1:
        # if learner wins, use pos_learner skill which is updated with them topics ;)
        # try:
        new_team_learner, _ = trueskill.rate([team_learner, team_content], ranks=[0, 0],
                                             weights=[learner_weights, content_weights])
        # except Exception:
        #     print()
    else:  # if the person is not engaged...
        # check if the winner is learner or content, uses the predicted skill representation
        difference = np.sum(team_mean_learner) - np.sum(team_mean_content)

        if difference > 0.:  # learner wins --> boring content
            new_team_learner, _ = trueskill.rate([team_learner, team_content], ranks=[0, 1],
                                                 weights=[learner_weights, content_weights])
        elif difference < 0.:  # learner loses --> intimidation
            # try:
            _, new_team_learner = trueskill.rate([team_content, team_learner], ranks=[0, 1],
                                                 weights=[content_weights, learner_weights])
            # except Exception:
            #     print()
        else:
            new_team_learner = team_learner

    return new_team_learner


def predict_and_model_truelearn_novel(events, user_model, draw_probability, draw_factor, topics_covered,
                                      positive_only, stats, init_skill, def_var, beta_sqr, tau, semantic_mapping,
                                      agg_func, is_pred_only, is_diluted, dil_factor, var_const, top_k_sr_topics,
                                      sr_func, is_tracking, is_topics, is_fixed, is_interest, decay_func, n_topics,
                                      threshold, start_event, q_random):
    random.seed(42)

    actual = []  # as the draw probability cant be zero
    predicted = []
    video_change = []
    # videos = []

    num_updates = []  # to record how many updates happened before prediction
    topics = []  # to track what topics are being used

    prev_label = None

    # if is_video:
    #     start_event = 0
    # # else:
    # start_event = 0

    if is_tracking:
        turns = 5
        tot_durations = []
    else:
        turns = 1
        tot_durations = None

    rel_topics = {}  # to track related topics
    current_vid = None

    for turn in range(turns):

        start_time = datetime.now()
        for idx, event in enumerate(events):
            #  calculate if the user is going to engage with this resource

            user_id, slug, vid_id, part_id, event_time, event_timeframe, topic_vec, label = decode_vector(event,
                                                                                                          n_topics)

            topic_dict = get_topic_dict(topic_vec)

            # setup trueskill environment
            if draw_probability == "static":
                _draw_probability = STATIC_DRAW_PROB  # population success rate
                # _draw_probability = 1.
            else:
                # compute novelty prob
                _draw_probability = float(np.mean(actual))
                _draw_probability = _draw_probability if _draw_probability > 0. else 0.000000000001  # cant be zero
                _draw_probability = _draw_probability if _draw_probability < 1. else 0.999999999999  # cant be one

            _draw_probability *= draw_factor

            trueskill.setup(mu=0.0, sigma=1 / 1000000000, beta=float(np.sqrt(beta_sqr)), tau=tau,
                            draw_probability=_draw_probability,
                            backend="mpmath")

            # track unique topic encountered
            topics_covered |= set(topic_dict.keys())

            # create_teams
            team_learner = tuple()
            learner_weights = tuple()
            orig_team_learner = tuple()

            # pos_team_learner = tuple()
            team_learner_mean_vec = list()
            team_learner_var_vec = list()
            team_mean_learner = list()

            team_content = tuple()
            content_weights = tuple()

            team_content_mean_vec = list()
            team_content_var_vec = list()
            team_mean_content = list()

            topic_seq = []
            tmp_updates = []

            _topic = None
            for topic, coverage in topic_dict.items():
                topic_seq.append(topic)

                if is_topics and topic not in user_model:

                    if semantic_mapping is not None:
                        # it is fresh
                        _rel_topics = get_related_skill_set(user_model, semantic_mapping, topic, top_k_sr_topics,
                                                            sr_func=sr_func)
                    else:
                        _rel_topics = {}

                    _topic = topic
                    rel_topics[topic] = {
                        "idx": idx,
                        "rel_topics": _rel_topics
                    }

                # get user skill rating
                tmp_learner_skill = user_model["mean"].get(topic, init_skill)
                tmp_learner_sd = np.sqrt(user_model["variance"].get(topic, def_var) + var_const)
                tmp_learner_f = float(user_model["updates"].get(topic, 0))
                tmp_learner_l_time = float(user_model["last_time"].get(topic, event_time))

                tmp_updates.append(tmp_learner_f)

                # for reinitialising learner if predict only
                orig_learner_skill = trueskill.Rating(mu=tmp_learner_skill, sigma=tmp_learner_sd)

                # if semantic truelearn and unobserved topic
                if semantic_mapping is not None and tmp_learner_skill == init_skill and tmp_learner_sd == np.sqrt(
                        def_var):
                    updated_learner = get_semantic_skill_inference(user_model, semantic_mapping, topic, init_skill,
                                                                   def_var, event_time,
                                                                   agg_func, top_k_sr_topics, sr_func)

                    tmp_learner_skill, tmp_learner_sd = updated_learner

                # apply interest decay
                if decay_func:
                    tmp_learner_skill = apply_interest_decay(tmp_learner_skill, event_time, tmp_learner_l_time,
                                                             decay_func)

                # used for prediction
                learner_skill = trueskill.Rating(mu=tmp_learner_skill, sigma=tmp_learner_sd)

                # if is_diluted:
                #     tmp_learner_skill, tmp_learner_sd = dilute_variance(orig_learner_skill.mu, orig_learner_skill.sigma,
                #                                                         user_model, semantic_mapping, topic, dil_factor,
                #                                                         top_k_sr_topics, def_var)
                #     # if not predicting with this, replace with original values
                #     if not is_pred_only:
                #         learner_skill = trueskill.Rating(mu=orig_learner_skill.mu, sigma=orig_learner_skill.sigma)
                #
                #     # used for learning with diluted variance of semantically related new topic
                #     orig_learner_skill = trueskill.Rating(mu=tmp_learner_skill, sigma=tmp_learner_sd)

                # for reinitialising learner if predict only
                orig_team_learner += (orig_learner_skill,)

                # for prediction
                team_learner += (learner_skill,)
                # learner_weights += (coverage,)
                learner_weights += (1.,)

                team_mean_learner.append(learner_skill.mu)
                team_learner_mean_vec.append(tmp_learner_skill)
                team_learner_var_vec.append(np.square(tmp_learner_sd))

                # get skill coverage
                tmp_content_topic = coverage
                topic_cov = trueskill.Rating(mu=tmp_content_topic, sigma=1 / 1000000000)
                team_content += (topic_cov,)
                content_weights += (1.,)

                team_mean_content.append(topic_cov.mu)
                team_content_mean_vec.append(tmp_content_topic)
                team_content_var_vec.append(np.square(1 / 1000000000))

            # check if user engages
            # test_team_learer = deepcopy(team_learner)

            # pred_list = []
            # for i, topic in enumerate(topic_seq):
            #     tmp_team_learner = (team_learner[i],)
            #     tmp_team_content = (team_content[i],)

            # pred_learner = (team_learner[0],)
            # pred_content = (team_content[0],)
            # pred_learner_weights = (learner_weights[0],)
            # pred_content_weights = (content_weights[0],)

            if is_fixed or is_interest:
                # engage if player is better than content, hence sum quality rather than draw...
                pred_prob = team_sum_quality(np.array(team_learner_mean_vec),
                                             np.array(team_learner_var_vec),
                                             np.array(team_content_mean_vec),
                                             np.array(team_content_var_vec), np.sqrt(beta_sqr))
            else:
                pred_prob = trueskill.quality([team_learner, team_content], weights=[learner_weights, content_weights])

            if q_random and idx == 0:  # means first event has to be randomly predicted
                pred_prob = random.uniform(0, 1)
            # if user engages, update the model
            label = event[-1]

            if is_interest:
                orig_label = label
                label = 1

            # if idx ==600:
            #     print()

            # if label is negative and setting is positive only, skip updating
            if positive_only and label != 1:
                pass
            else:
                if is_pred_only:
                    team_learner = orig_team_learner

                if is_fixed or is_interest:
                    new_team_learner = update_truelearn_fixed(label, team_learner, team_content)
                else:
                    new_team_learner = update_truelearn_novel(label, team_learner, team_content, learner_weights,
                                                              content_weights, team_mean_learner, team_mean_content)

                for _idx, topic in enumerate(topic_seq):
                    user_model["mean"][topic], user_model["variance"][
                        topic] = new_team_learner[_idx].mu, new_team_learner[_idx].sigma ** 2
                    user_model["updates"][topic] = user_model["updates"].get(topic, 0) + 1
                    user_model["last_time"][topic] = event_time

            # if not first element, calculate accuracy
            if idx >= start_event:

                if is_interest:
                    label = orig_label

                if label != prev_label:
                    stats["change_label"] += 1

                actual.append(label)
                predicted.append(pred_prob)

            # if current_vid is None:
            #     video_change.append(1)
            #     current_vid = (slug, vid_id)
            if current_vid != (slug, vid_id):  # if the video is changing
                stats["vid_changes"] += 1
                video_change.append(1)
                current_vid = (slug, vid_id)  # reassign new video
                if label == int(pred_prob >= threshold):  # if the first prediction is correct
                    stats["vid_pos_changes"] += 1
            else:
                video_change.append(0)

            # videos.append(current_vid)

            num_updates.append(tmp_updates)  # to track the number of updates
            topics.append(topic_seq)
            prev_label = label

        if is_tracking:
            stop_time = datetime.now()
            difference_in_milliseconds = float((stop_time - start_time) / timedelta(milliseconds=1))
            tot_durations.append(difference_in_milliseconds)

    stats = dict(stats)
    stats["total_duration"] = tot_durations
    stats["rel_topics"] = rel_topics

    # stats["videos"] = videos
    stats["video_change"] = video_change

    stats["topics"] = topics
    stats["num_updates"] = num_updates

    return actual, predicted, user_model, topics_covered, stats


def get_interest_decay_func(type, factor):
    """
    # equations from: https://link.springer.com/article/10.1007/s11227-020-03266-2
    """

    if type == "short":
        return lambda t_delta: min(2 / (1 + np.exp(factor * t_delta)), 1.)
    else:
        return lambda t_delta: min(np.exp(-factor * t_delta), 1.)


def truelearn_novel_model(records, init_skill=0., def_var=None, tau=0., beta_sqr=0., threshold=0.5,
                          draw_probability="static", draw_factor=.1, positive_only=False, semantic_mapping=None,
                          tracking=False, agg_func="raw", is_pred_only=False, is_diluted=False, dil_factor=1.0,
                          var_const=0., top_k_sr_topics=-1, sr_func="raw", is_topics=False, is_fixed=False,
                          is_interest=False, decay_func=None, start_event=0):
    """This model calculates trueskill given all positive skill using the real trueskill factorgraph.
    Args:
        records [[val]]: list of vectors for each event of the user. Format of vector
            [user_id, slug, vid_id, part_id, timeframe, ... topic_vector ..., label]

    Returns:
        accuracy (float): accuracy for all observations
        concordance ([bool]): the concordance between actual and predicted values
    """

    # if is_video:
    #     records = get_videowise_data(records, is_video)
    #     num_records = float(len(records))
    # else: # if partwise
    num_records = float(len(records))

    if num_records <= 1:
        return 0., [], int(num_records), False

    user_model = {
        "mean": {},
        "variance": {},
        "updates": {},
        "last_time": {}
    }

    topics_covered = set()

    # actual = []
    # predicted = []

    stats = defaultdict(int)

    if not (semantic_mapping is None or type(semantic_mapping) == dict):  # convert broadcast variable to dict
        semantic_mapping = semantic_mapping.value

    # for every event
    (actual, pred_probs, user_model, topics_covered, stats) = predict_and_model_truelearn_novel(records, user_model,
                                                                                                draw_probability,
                                                                                                draw_factor,
                                                                                                topics_covered,
                                                                                                positive_only, stats,
                                                                                                init_skill,
                                                                                                def_var, beta_sqr, tau,
                                                                                                semantic_mapping,
                                                                                                agg_func, is_pred_only,
                                                                                                is_diluted, dil_factor,
                                                                                                var_const,
                                                                                                top_k_sr_topics,
                                                                                                sr_func, tracking,
                                                                                                is_topics, is_fixed,
                                                                                                is_interest,
                                                                                                decay_func, 5,
                                                                                                threshold, start_event)

    stats["num_topics"] = len(topics_covered)

    pred_probs = np.array(pred_probs).astype("float64")
    actual = np.array(actual).astype("int")

    accuracy, precision, recall, f1, roc_score, pr_score, stats = get_summary_stats(actual, pred_probs, num_records,
                                                                                    stats=stats,
                                                                                    user_model=user_model,
                                                                                    threshold=threshold)

    return accuracy, precision, recall, f1, roc_score, pr_score, int(num_records), stats


def compute_quality_based_predictions(events, mapping_b, start_event, n_topics):
    pred_probs = []
    turns = 1

    for turn in range(turns):
        for idx, event in enumerate(events):
            if idx < start_event:
                continue

            #  calculate if the user is going to engage with this resource
            user_id, slug, vid_id, part_id, event_time, event_timeframe, topic_vec, label = decode_vector(event,
                                                                                                          n_topics)

            pred_probs.append(mapping_b.value["all"].get(slug, .5))

    pred_probs = np.array(pred_probs).astype("float64")

    return pred_probs
