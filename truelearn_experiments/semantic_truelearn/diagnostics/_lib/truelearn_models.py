import datetime
from collections import defaultdict
import numpy as np
import trueskill

from analyses.truelearn_experiments.trueknowledge_recommender_models import dilute_variance, \
    get_semantic_skill_inference
from analyses.truelearn_experiments.utils import get_summary_stats, get_topic_dict


def _print_team_stats(team, team_name, debug_records, is_content=False):
    for idx, player in enumerate(team):
        i = idx + 1
        mean, std = player.mu, player.sigma

        _mean_title = "team_{}_topic_{}_mean".format(team_name, i)
        debug_records[_mean_title] = mean
        print("{}: {}".format(_mean_title, mean))

        if not is_content:
            _std_title = "team_{}_topic_{}_std".format(team_name, i)
            debug_records[_std_title] = std
            print("{}: {}".format(_std_title, std))

    return debug_records


def predict_and_model_truelearn_novel(events, user_model, draw_probability, draw_factor, topics_covered, threshold,
                                      positive_only, stats, init_skill, def_var, beta_sqr, tau, semantic_mapping,
                                      agg_func, is_pred_only, is_diluted, dil_factor, debug):
    actual = []  # as the draw probability cant be zero
    predicted = []
    prev_label = None
    if debug:
        debug_records = []

    # if is_video:
    #     start_event = 0
    # else:
    start_event = 1

    for idx, event in enumerate(events):

        if debug:
            debug_record = {
                "init_skill": init_skill,
                "beta_sqr": beta_sqr,
                "tau": tau
            }
            print("init_skill: {}".format(init_skill))
            print("beta_sqr: {}".format(beta_sqr))
            print("tau: {}".format(tau))

        # decompose event record

        #  calculate if the user is going to engage with this resource
        topic_vec = event[5:-1]
        topic_dict = get_topic_dict(topic_vec)

        # setup trueskill environment
        if draw_probability == "static":
            _draw_probability = float(0.5932538086581619)  # population success rate
            # _draw_probability = 1.
        else:
            # compute novelty prob
            _draw_probability = float(np.mean(actual))
            _draw_probability = _draw_probability if _draw_probability > 0. else 0.000000000001  # cant be zero
            _draw_probability = _draw_probability if _draw_probability < 1. else 0.999999999999  # cant be one

        if debug:
            debug_record["draw_probability"] = _draw_probability
            print("draw_probability: {}".format(_draw_probability))

        _draw_probability *= draw_factor

        if debug:
            debug_record["factor_draw_probability"] = _draw_probability
            print("factor_draw_probability: {}".format(_draw_probability))

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
        team_mean_learner = []

        team_content = tuple()
        content_weights = tuple()
        team_mean_content = []

        topic_seq = []

        for topic, coverage in topic_dict.items():
            topic_seq.append(topic)
            # get user skill rating
            tmp_learner_skill = user_model["mean"].get(topic, init_skill)
            tmp_learner_sd = np.sqrt(user_model["variance"].get(topic, def_var))

            orig_learner_skill = trueskill.Rating(mu=tmp_learner_skill, sigma=tmp_learner_sd)

            # if semantic truelearn and unobserved topic
            if semantic_mapping is not None and tmp_learner_skill == init_skill and tmp_learner_sd == np.sqrt(def_var):
                updated_learner = get_semantic_skill_inference(user_model, semantic_mapping, topic, init_skill, def_var,
                                                               agg_func)

                tmp_learner_skill, tmp_learner_sd = updated_learner

            # used for prediction
            learner_skill = trueskill.Rating(mu=tmp_learner_skill, sigma=tmp_learner_sd)

            if is_diluted:
                tmp_learner_skill, tmp_learner_sd = dilute_variance(orig_learner_skill.mu, orig_learner_skill.sigma,
                                                                    user_model, semantic_mapping, topic, dil_factor)

                # used for learning with diluted variance of semantically related new topic
                orig_learner_skill = trueskill.Rating(mu=tmp_learner_skill, sigma=tmp_learner_sd)

                if not is_pred_only:
                    learner_skill = trueskill.Rating(mu=tmp_learner_skill, sigma=tmp_learner_sd)

            # for learning as well if not predict only
            orig_team_learner += (orig_learner_skill,)

            # for prediction
            team_learner += (learner_skill,)
            # learner_weights += (coverage,)
            learner_weights += (1.,)

            team_mean_learner.append(learner_skill.mu)

            # get skill coverage
            tmp_content_topic = coverage
            topic_cov = trueskill.Rating(mu=tmp_content_topic, sigma=1 / 1000000000)
            team_content += (topic_cov,)
            content_weights += (1.,)

            team_mean_content.append(tmp_content_topic)

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
        if debug:
            debug_records = _print_team_stats(team_learner, "learner_pred", debug_records)
            debug_records = _print_team_stats(team_content, "content_pred", debug_records, True)

        pred_prob = trueskill.quality([team_learner, team_content], weights=[learner_weights, content_weights])
        prediction = int(pred_prob >= threshold)
        # pred_list.append(prediction)

        # prediction = int(sum(pred_list) == len(topic_seq))
        # if user engages, update the model
        label = event[-1]

        # if idx ==600:
        #     print()

        # if label is negative and setting is positive only, skip updating
        if positive_only and label != 1:
            pass
        else:
            # try:
            # if predict only, change back to original repr
            if is_pred_only:
                team_learner = orig_team_learner

            # do pairwise learning
            # updated_team_learner = tuple()

            # for i, topic in enumerate(topic_seq):
            #     tmp_team_learner = (team_learner[i],)
            #     tmp_team_content = (team_content[i],)
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

            # updated_team_learner += new_team_learner

            for _idx, topic in enumerate(topic_seq):
                user_model["mean"][topic], user_model["variance"][
                    topic] = new_team_learner[_idx].mu, new_team_learner[_idx].sigma ** 2
                user_model["updates"][topic] = user_model["updates"].get(topic, 0) + 1

        # if not first element, calculate accuracy
        if idx >= start_event:
            if label != prev_label:
                stats["change_label"] += 1

            actual.append(label)
            predicted.append(prediction)

        prev_label = label

    return actual, predicted, user_model, topics_covered, stats


def truelearn_novel_model(records, init_skill=0., def_var=None, tau=0., beta_sqr=0., threshold=0.5,
                          draw_probability="static", draw_factor=.1, positive_only=False, semantic_mapping=None,
                          tracking=False, agg_func="raw", is_pred_only=False, is_diluted=False, dil_factor=1.0,
                          debug=False):
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

    times = []

    user_model = {
        "mean": {},
        "variance": {},
        "updates": {}
    }

    topics_covered = set()

    # actual = []
    # predicted = []

    stats = defaultdict(int)

    start_time = datetime.datetime.now()
    # for every event

    (actual, predicted, user_model, topics_covered, stats) = predict_and_model_truelearn_novel(records, user_model,
                                                                                               draw_probability,
                                                                                               draw_factor,
                                                                                               topics_covered,
                                                                                               threshold, positive_only,
                                                                                               stats, init_skill,
                                                                                               def_var, beta_sqr, tau,
                                                                                               semantic_mapping,
                                                                                               agg_func, is_pred_only,
                                                                                               is_diluted, dil_factor,
                                                                                               debug)

    # actual.append(bundle_actual)
    # predicted.append(bunlde_predicted)

    stats = dict(stats)

    stats["num_topics"] = len(topics_covered)

    end_time = datetime.datetime.now()
    time_diff = end_time - start_time

    times.append(time_diff)

    if tracking:
        time_cost = (np.mean(times) / num_records).microseconds
        return time_cost, actual, predicted, int(num_records)

    accuracy, precision, recall, f1, stats = get_summary_stats(actual, predicted, num_records, stats=stats,
                                                               user_model=user_model)

    return accuracy, precision, recall, f1, int(num_records), stats
