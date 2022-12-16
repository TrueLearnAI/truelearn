import itertools
from collections import defaultdict

import numpy as np

from analyses.truelearn_experiments.utils import get_topic_dict, get_summary_stats, decode_vector, apply_interest_decay


def train_joint_probability(event_distribution, engagement, pguess=0.25, pfail=0., joint_criteria="and"):
    """

    Args:
        event_distribution ({int: float}): topic id, prob of knowing of each topic of the user
        engagement (bool):
        pguess (float):
        pfail (float):

    Returns:

    """
    # user_model is a vector representing the probability of the user knowing each topic
    # topics is a vector representing the binary coverage of the content
    # bool_eng is a boolean related to engagement.
    # pguess is the probability of not having the appropriate knowledge but still being engaged
    # pfail is the probability of having the appropriate knowledge but not being engaged

    # IMPORTANT: 1) with this model we would assume that it is equally likely that the learner
    # knows or doesn't know a topic a priori. This is, we need to initialise skills to 0.5, not to 0.
    # 2) The model assumes that once someone learns something completely (prob = 1) you can't unlearn
    # it. So if a skill is 1 it won't move.

    # we should perhaps set pfail higher than pguess. It should be quite probable than someone
    # that does have the knowledge is not engaged.

    # get skills used
    relevant_skills = list(event_distribution.keys())
    relevant_skills_probs = list(event_distribution.values())
    k = len(relevant_skills_probs)

    # get table of binary combinations for bernouilli variables
    truthTable = list(itertools.product([0, 1], repeat=k))
    truthTable = np.array(truthTable)

    # number of binary combinations
    ncomb = len(truthTable)
    # vector that keeps the result of computing AND operator for the binary combinations
    and_vector = np.zeros(ncomb)

    # Incorporate pguess and pfail to bool_eng
    is_engaged_prob = (engagement * ((1 - pfail) / (1 - pfail + pguess)) +
                       (not (engagement)) * ((pfail) / (1 + pfail - pguess)))

    # compute AND
    for i in range(ncomb):
        if joint_criteria == "and":
            and_vector[i] = np.prod(truthTable[i])
        else:
            and_vector[i] = (np.sum(truthTable[i]) > 0.)  # to make an OR vector

    table_probs = np.zeros((ncomb, k + 1))
    # compute table of probabilities for binary skills
    for i in range(k):
        for j in range(ncomb):
            table_probs[j, i] = truthTable[j, i] * relevant_skills_probs[i] + (not (truthTable[j, i])) * (
                    1 - relevant_skills_probs[i])

    # compute table of probabilities for AND
    for j in range(ncomb):
        table_probs[j, k] = and_vector[j] * is_engaged_prob + (not (and_vector[j])) * (1 - is_engaged_prob)

    new_skill = np.zeros(k)
    # for each variable we compute the update
    for i in range(k):
        ptrue = 0
        pfalse = 0
        for j in range(ncomb):
            # the update is computed as a product of all other messages (except the one we are considering) summing up over all variables
            table_probs_tmp = np.delete(table_probs[j], i)
            product = np.prod(table_probs_tmp)
            ptrue = ptrue + (truthTable[j, i] * product)
            pfalse = pfalse + ((not (truthTable[j, i])) * product)

        # compute update
        update = ptrue / (ptrue + pfalse)
        # multiply by prior
        new_skill[i] = (relevant_skills_probs[i] * update) / (
                relevant_skills_probs[i] * update + (1 - relevant_skills_probs[i]) * (1 - update))

    new_user_model = {k: new_skill[idx] for idx, k in enumerate(relevant_skills)}

    return new_user_model


def predict_joint_probability(event_distribution, pguess=0.25, pfail=0.0, joint_criteria="and"):
    """

    Args:
        event_distribution:
        topic_dict:
        pguess:
        pfail:
        threshold:

    Returns:
        bool: True if prob > threshold, else False
    """

    # get skills used
    relevant_skills = list(event_distribution.keys())
    relevant_skills_probs = list(event_distribution.values())
    k = len(relevant_skills_probs)

    # get table of binary combinations for bernouilli variables
    truthTable = list(itertools.product([0, 1], repeat=k))
    truthTable = np.array(truthTable)

    # number of binary combinations
    ncomb = len(truthTable)
    # vector that keeps the result of computing AND operator for the binary combinations
    and_vector = np.zeros(ncomb)

    # compute AND
    for i in range(ncomb):
        if joint_criteria == "and":
            and_vector[i] = np.prod(truthTable[i])
        else:
            and_vector[i] = (np.sum(truthTable[i]) > 0.)

    table_probs = np.zeros((ncomb, k))
    # compute table of probabilities for binary skills
    for i in range(k):
        for j in range(ncomb):
            # ipdb.set_trace()
            table_probs[j, i] = (truthTable[j, i] * relevant_skills_probs[i] +
                                 (not (truthTable[j, i])) * (1 - relevant_skills_probs[i]))

    ptrue = 0
    pfalse = 0

    for j in range(ncomb):
        product = np.prod(table_probs[j])
        ptrue = ptrue + (and_vector[j] * product)
        pfalse = pfalse + ((not (and_vector[j])) * product)

    # compute prob
    prob_tmp = ptrue / (ptrue + pfalse)

    # Incorporate pguess and pfail to bool_eng
    prob = prob_tmp * ((1 - pfail) / (1 - pfail + pguess)) + (1 - prob_tmp) * ((pfail) / (1 + pfail - pguess))
    return prob


def predict_and_model_kt_model(records, user_model, init_certainty, pguess, pfail, stats, topics_covered, decay_func,
                               is_interest, positive_only, start_event):
    actual = []  # as the draw probability cant be zero
    predicted = []
    prev_label = None

    rel_topics = {}

    for idx, event in enumerate(records):
        #  calculate if the user is going to engage with this resource
        user_id, slug, vid_id, part_id, event_time, event_timeframe, topic_vec, label = decode_vector(event)

        topic_dict = get_topic_dict(topic_vec)

        # track unique topic encountered
        topics_covered |= set(topic_dict.keys())

        # check if user engages
        temp_user_model = {}

        for topic in topic_dict:
            tmp_learner_skill = user_model["mean"].get(topic, init_certainty)

            # tmp_learner_f = float(user_model["updates"].get(topic, 0))
            tmp_learner_l_time = float(user_model["last_time"].get(topic, event_time))

            # apply interest decay
            if decay_func:
                tmp_learner_skill = apply_interest_decay(tmp_learner_skill, event_time, tmp_learner_l_time, decay_func)

            temp_user_model[topic] = tmp_learner_skill

        pred_prob = float(predict_joint_probability(temp_user_model, pguess=pguess, pfail=pfail))

        if is_interest:
            orig_label = label
            label = 1

        # if label is negative and setting is positive only, skip updating
        if positive_only and label != 1:
            pass
        else:
            # update if label is positive or negative
            temp_label = bool(label)

            temp_user_model = train_joint_probability(temp_user_model, temp_label, pguess=pguess, pfail=pfail)

            label = int(label)

            for topic, p_know in temp_user_model.items():
                user_model["mean"][topic] = p_know
                user_model["updates"][topic] = float(user_model["updates"].get(topic, 0)) + 1
                user_model["last_time"][topic] = event_time

        # if not first element, calculate accuracy
        if idx >= start_event:

            if is_interest:
                label = orig_label

            if label != prev_label:
                stats["change_label"] += 1

            actual.append(label)
            predicted.append(pred_prob)

        prev_label = label

    stats = dict(stats)

    # num topics
    stats["num_topics"] = len(topics_covered)
    stats["user_model"] = None

    return actual, predicted, user_model, topics_covered, stats


def knowledge_tracing_model(records, init_certainty=0.5, tau=None, beta_sqr=None, threshold=0.5, positive_only=True,
                            is_interest=False, decay_func=None, start_event=0):
    """This model calculates trueskill given all positive skill.
    Args:
        records [[val]]: list of vectors for each event of the user. Format of vector
            [session, time, timeframe_id, topic_id, topic_cov ..., label]
        init_certainty (float): initial uncertainty
        tau (float): p_guess
        beta_sqr (float): p_fail
        engage_fun: function that estimates engagement probability
        threshold (float): engagement threshold

    Returns:
        accuracy (float): accuracy for all observations
        concordance ([bool]): the concordance between actual and predicted values
    """
    if tau is None:
        tau = 0.

    if beta_sqr is None:
        beta_sqr = 0.

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
    stats = defaultdict(int)

    (actual, pred_probs, user_model, topics_covered, stats) = predict_and_model_kt_model(records, user_model,
                                                                                         init_certainty, tau, beta_sqr,
                                                                                         stats, topics_covered,
                                                                                         decay_func, is_interest,
                                                                                         positive_only, start_event)

    pred_probs = np.array(pred_probs).astype("float64")
    actual = np.array(actual).astype("int")

    accuracy, precision, recall, f1, roc_score, pr_score, stats = get_summary_stats(actual, pred_probs, num_records,
                                                                                    threshold=threshold,
                                                                                    stats=stats, user_model=user_model)

    return accuracy, precision, recall, f1, roc_score, pr_score, int(num_records), stats
