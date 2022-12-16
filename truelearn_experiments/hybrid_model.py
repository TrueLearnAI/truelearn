import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from analyses.truelearn_experiments.knowledge_tracing_model import predict_joint_probability
from analyses.truelearn_experiments.trueknowledge_recommender_models import predict_and_model_truelearn_novel, \
    compute_quality_based_predictions
from analyses.truelearn_experiments.utils import get_summary_stats, _cdf


def get_werigted_probs(k_preds, i_preds, kp):
    return (kp * k_preds) + (1 - kp) * i_preds


def _compute_probability(k_pred, i_pred, type="and"):
    assert len(k_pred) == len(i_pred)

    length = len(k_pred)

    final_preds = np.zeros(length)

    for idx in range(length):
        probs = {
            "knowledge": k_pred[idx],
            "interest": i_pred[idx]
        }

        final_preds[idx] = predict_joint_probability(probs, pguess=.0, pfail=.0, joint_criteria=type)

    return final_preds


def _compute_weighted_probability(k_pred_probs, i_pred_probs, know_prob=1.0):
    assert 0. <= know_prob <= 1.
    assert len(k_pred_probs) == len(i_pred_probs)

    pred_probs = get_werigted_probs(k_pred_probs, i_pred_probs, know_prob)

    return pred_probs


def _compute_accuracy_probability(k_pred_probs, i_pred_probs, actual, threshold=.5):
    assert len(k_pred_probs) == len(i_pred_probs) == len(actual)

    # turn everything to boolean
    k_pred_labels = (k_pred_probs >= threshold).astype("int")
    i_pred_labels = (i_pred_probs >= threshold).astype("int")
    actual = actual.astype("int")
    # for range from length 1 to full list

    final_probs = np.zeros(len(actual))
    weights = np.zeros(len(actual))

    for idx in range(len(actual)):
        # get past events
        tmp_k = k_pred_labels[:idx]
        tmp_i = i_pred_labels[:idx]
        tmp_act = actual[:idx]

        # if first event, no info,
        if len(tmp_k) == len(tmp_i) == len(tmp_act) == 0:
            acc_k = acc_i = 0.
        else:
            acc_k = accuracy_score(tmp_act, tmp_k)
            acc_i = accuracy_score(tmp_act, tmp_i)

        if acc_k == acc_i == 0.:
            kp = .5
        else:
            kp = acc_k / (acc_k + acc_i)

        final_probs[idx] = get_werigted_probs(k_pred_probs[idx], i_pred_probs[idx], kp)
        weights[idx] = kp
    return final_probs, weights


def _compute_f1_probability(k_pred_probs, i_pred_probs, actual, threshold=.5):
    assert len(k_pred_probs) == len(i_pred_probs) == len(actual)

    # turn everything to boolean
    k_pred_labels = (k_pred_probs >= threshold).astype("int")
    i_pred_labels = (i_pred_probs >= threshold).astype("int")
    actual = actual.astype("int")
    # for range from length 1 to full list

    final_probs = np.zeros(len(actual))
    weights = np.zeros(len(actual))

    for idx in range(len(actual)):
        # get past events
        tmp_k = k_pred_labels[:idx]
        tmp_i = i_pred_labels[:idx]
        tmp_act = actual[:idx]

        # if first event, no info,
        if len(tmp_k) == len(tmp_i) == len(tmp_act) == 0:
            acc_k = acc_i = 0.
        else:
            acc_k = f1_score(tmp_act, tmp_k)
            acc_i = f1_score(tmp_act, tmp_i)

        if acc_k == acc_i == 0.:
            kp = .5
        else:
            kp = acc_k / (acc_k + acc_i)

        final_probs[idx] = get_werigted_probs(k_pred_probs[idx], i_pred_probs[idx], kp)
        weights[idx] = kp

    return final_probs, weights


def compute_logistic_probability(k_pred_probs, i_pred_probs, actual):
    from sklearn.linear_model import SGDClassifier

    assert len(k_pred_probs) == len(i_pred_probs) == len(actual)

    # instantiate stochastic logistic classifier
    sgd_logistic = SGDClassifier(loss="log", l1_ratio=0., fit_intercept=False, n_jobs=1, random_state=42)

    final_probs = []
    weights = []

    for idx, (k, i, act) in enumerate(zip(k_pred_probs, i_pred_probs, actual)):
        if idx == 0:
            sgd_logistic.partial_fit([[.0, .0]], [0], classes=np.unique([0, 1]))  # initial training
            sgd_logistic.coef_ = np.array([[.5, .5]])

        weights.append(sgd_logistic.coef_.tolist()[0])

        x = [k, i]
        tmp_pred = sgd_logistic.predict_proba([x])
        sgd_logistic.partial_fit([x], [act])

        # add probability p(1) as p(0) + p(1) = 1.
        final_probs.append(tmp_pred[0][1])

    return np.array(final_probs).astype("float64"), np.array(weights)


def compute_perceptron_prediction(k_pred_probs, i_pred_probs, actual):
    from sklearn.linear_model import SGDClassifier

    assert len(k_pred_probs) == len(i_pred_probs) == len(actual)

    # instantiate stochastic logistic classifier
    sgd_perceptron = SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None,
                                   fit_intercept=False,
                                   n_jobs=1, random_state=42)

    final_preds = []
    weights = []

    for idx, (k, i, act) in enumerate(zip(k_pred_probs, i_pred_probs, actual)):
        if idx == 0:
            sgd_perceptron.partial_fit([[.0, .0]], [0], classes=np.unique([0, 1]))  # initial training
            sgd_perceptron.coef_ = np.array([[.5, .5]])

        weights.append(sgd_perceptron.coef_.tolist()[0])

        x = [k, i]
        tmp_pred = sgd_perceptron.predict([x])
        sgd_perceptron.partial_fit([x], [act])

        # add probability p(1) as p(0) + p(1) = 1.
        final_preds.append(tmp_pred[0])

    return np.array(final_preds).astype("float64"), np.array(weights)


def compute_pagg_prediction(k_pred_probs, i_pred_probs, actual):
    from sklearn.linear_model import PassiveAggressiveClassifier

    assert len(k_pred_probs) == len(i_pred_probs) == len(actual)

    # instantiate stochastic logistic classifier
    pass_agg_model = PassiveAggressiveClassifier(fit_intercept=False, n_jobs=1, random_state=42)

    final_preds = []
    weights = []

    for idx, (k, i, act) in enumerate(zip(k_pred_probs, i_pred_probs, actual)):
        if idx == 0:
            pass_agg_model.partial_fit([[.0, .0]], [0], classes=np.unique([0, 1]))  # initial training
            pass_agg_model.coef_ = np.array([[.5, .5]])

        weights.append(pass_agg_model.coef_.tolist()[0])

        x = [k, i]
        tmp_pred = pass_agg_model.predict([x])
        pass_agg_model.partial_fit([x], [act])

        # add probability p(1) as p(0) + p(1) = 1.
        final_preds.append(tmp_pred[0])

    return np.array(final_preds).astype("float64"), np.array(weights)


def _calculate_sum_prediction(mu_k, sd_k, k, mu_i, sd_i, i, mu_b, sd_b, b, threshold, mu_q=0., sd_q=0., q=0.):
    difference = (mu_k * k) + (mu_i * i) + (mu_q * q) + (mu_b * b) - threshold
    std = np.sqrt(
        ((sd_k ** 2) * (k ** 2)) + ((sd_i ** 2) * (i ** 2)) + ((sd_q ** 2) * (q ** 2)) + ((sd_b ** 2) * (b ** 2)))
    return float(_cdf(difference, 0, std))


def get_quality_mapping(mapping_path):
    with open(mapping_path) as inf:
        mapping = json.load(inf)

    new_mapping = {
        "train": {}, "test": {}, "all": {}

    }

    for t, m in mapping.items():
        for k, v in m.items():
            duration = v["duration"]
            prediction = v["prediction"]

            pred_duration = duration * prediction

            new_mapping[t][k] = min(pred_duration / 300., 1.)

    new_mapping["all"] = {**new_mapping["train"], **new_mapping["test"]}

    return new_mapping


def compute_truelearn_prediction(k_pred_probs, i_pred_probs, actual, know_prob, tau, threshold, is_greedy):
    import trueskill

    assert len(k_pred_probs) == len(i_pred_probs) == len(actual)

    final_preds = []
    weights = []

    # model
    current_weights = {"mean": {"k": .5, "i": .5,
                                "b": .0},
                       "sigma": {"k": know_prob, "i": know_prob,
                                 "b": know_prob}}

    trueskill.setup(mu=0.0, sigma=1 / 1000000000, beta=1, tau=tau, draw_probability=1 / 1000000000, backend="mpmath")

    for idx, (k, i, act) in enumerate(zip(k_pred_probs, i_pred_probs, actual)):
        mu_k, mu_i, mu_b = current_weights["mean"]["k"], current_weights["mean"]["i"], current_weights["mean"]["b"]
        sd_k, sd_i, sd_b = current_weights["sigma"]["k"], current_weights["sigma"]["i"], current_weights["sigma"]["b"]

        weights.append([mu_k, mu_i])

        tmp_pred = _calculate_sum_prediction(mu_k, sd_k, k, mu_i, sd_i, i, mu_b, sd_b, 1, threshold)
        final_preds.append(tmp_pred)

        # if prediction is correct and greedy,
        if is_greedy and int(tmp_pred >= threshold) == act:
            # dont train.
            continue

        # train
        team_experts = (trueskill.Rating(mu=mu_k, sigma=sd_k), trueskill.Rating(mu=mu_i, sigma=sd_i),
                        trueskill.Rating(mu=mu_b, sigma=sd_b))

        team_threshold = (trueskill.Rating(mu=.5, sigma=1 / 1000000000),)

        if act == 1:  # weights need to be larger than threshold
            new_team_experts, _ = trueskill.rate([team_experts, team_threshold], weights=[(k, i, 1), (1,)],
                                                 ranks=[0, 1])
        else:
            new_team_experts, _ = trueskill.rate([team_experts, team_threshold], weights=[(k, i, 1), (1,)],
                                                 ranks=[1, 0])

        # set skills
        new_mu_k, new_mu_i, new_mu_b = new_team_experts[0].mu, new_team_experts[1].mu, new_team_experts[2].mu
        new_sd_k, new_sd_i, new_sd_b = new_team_experts[0].sigma, new_team_experts[1].sigma, new_team_experts[2].sigma

        current_weights["mean"]["k"], current_weights["mean"]["i"], current_weights["mean"][
            "b"] = new_mu_k, new_mu_i, new_mu_b
        current_weights["sigma"]["k"], current_weights["sigma"]["i"], current_weights["sigma"][
            "b"] = new_sd_k, new_sd_i, new_sd_b

    return np.array(final_preds).astype("float64"), np.array(weights)


def compute_truelearn_pink(k_pred_probs, i_pred_probs, q_pred_probs, actual, know_prob, tau, threshold, is_greedy):
    import trueskill

    assert len(k_pred_probs) == len(i_pred_probs) == len(q_pred_probs) == len(actual)

    final_preds = []
    weights = []

    # model
    current_weights = {"mean": {"k": .05, "i": .05, "q": .9,
                                "b": .0},
                       "sigma": {"k": know_prob, "i": know_prob, "q": know_prob,
                                 "b": know_prob}}

    trueskill.setup(mu=0.0, sigma=1 / 1000000000, beta=1, tau=tau, draw_probability=1 / 1000000000, backend="mpmath")

    for idx, (k, i, q, act) in enumerate(zip(k_pred_probs, i_pred_probs, q_pred_probs, actual)):
        mu_k, mu_i, mu_q, mu_b = current_weights["mean"]["k"], current_weights["mean"]["i"], current_weights["mean"][
            "q"], current_weights["mean"]["b"]
        sd_k, sd_i, sd_q, sd_b = current_weights["sigma"]["k"], current_weights["sigma"]["i"], current_weights["sigma"][
            "q"], current_weights["sigma"]["b"]

        weights.append([mu_k, mu_i, mu_q])

        tmp_pred = _calculate_sum_prediction(mu_k, sd_k, k, mu_i, sd_i, i, mu_b, sd_b, 1, threshold, mu_q=mu_q,
                                             sd_q=sd_q, q=q)
        final_preds.append(tmp_pred)

        # if prediction is correct and greedy,
        if is_greedy and int(tmp_pred >= threshold) == act:
            # dont train.
            continue

        # train
        team_experts = (trueskill.Rating(mu=mu_k, sigma=sd_k), trueskill.Rating(mu=mu_i, sigma=sd_i),
                        trueskill.Rating(mu=mu_q, sigma=sd_q),
                        trueskill.Rating(mu=mu_b, sigma=sd_b))

        team_threshold = (trueskill.Rating(mu=.5, sigma=1 / 1000000000),)

        if act == 1:  # weights need to be larger than threshold
            new_team_experts, _ = trueskill.rate([team_experts, team_threshold], weights=[(k, i, q, 1), (1,)],
                                                 ranks=[0, 1])
        else:
            new_team_experts, _ = trueskill.rate([team_experts, team_threshold], weights=[(k, i, q, 1), (1,)],
                                                 ranks=[1, 0])

        # set skills
        new_mu_k, new_mu_i, new_mu_q, new_mu_b = (
            new_team_experts[0].mu, new_team_experts[1].mu, new_team_experts[2].mu,
            new_team_experts[3].mu)
        new_sd_k, new_sd_i, new_sd_q, new_sd_b = (
            new_team_experts[0].sigma, new_team_experts[1].sigma, new_team_experts[2].sigma, new_team_experts[3].sigma)

        (current_weights["mean"]["k"], current_weights["mean"]["i"], current_weights["mean"]["q"],
         current_weights["mean"]["b"]) = new_mu_k, new_mu_i, new_mu_q, new_mu_b
        (current_weights["sigma"]["k"], current_weights["sigma"]["i"], current_weights["sigma"]["q"],
         current_weights["sigma"]["b"]) = new_sd_k, new_sd_i, new_sd_q, new_sd_b

    return np.array(final_preds).astype("float64"), np.array(weights)


def compute_hybrid_probabilities(k_pred_probs, i_pred_probs, prob_combine_type, actual=None, know_prob=None, tau=0.,
                                 threshold=.5, weighted=False, quality_preds=None):
    if weighted:
        final_preds, weights = compute_truelearn_pink(k_pred_probs, i_pred_probs, quality_preds, actual, know_prob, tau,
                                                      threshold, False)

        return final_preds, weights

    if prob_combine_type == "and":  # and
        final_preds = _compute_probability(k_pred_probs, i_pred_probs, type="and")
        weights = None
    elif prob_combine_type == "or":  # or
        final_preds = _compute_probability(k_pred_probs, i_pred_probs, type="or")
        weights = None
    elif prob_combine_type == "weight":  # weights
        assert know_prob is not None
        final_preds = _compute_weighted_probability(k_pred_probs, i_pred_probs, know_prob=know_prob)
        weights = None
    elif prob_combine_type == "acc_weight":
        final_preds, weights = _compute_accuracy_probability(k_pred_probs, i_pred_probs, actual, threshold=threshold)
    elif prob_combine_type == "f1_weight":
        final_preds, weights = _compute_f1_probability(k_pred_probs, i_pred_probs, actual, threshold=threshold)
    elif prob_combine_type == "meta-logistic":
        final_preds, weights = compute_logistic_probability(k_pred_probs, i_pred_probs, actual)
    elif prob_combine_type == "meta-perceptron":
        final_preds, weights = compute_perceptron_prediction(k_pred_probs, i_pred_probs, actual)
    elif prob_combine_type == "meta-truelearn":
        final_preds, weights = compute_truelearn_prediction(k_pred_probs, i_pred_probs, actual, know_prob, tau,
                                                            threshold, False)
    elif prob_combine_type == "meta-truelearn-greedy":
        final_preds, weights = compute_truelearn_prediction(k_pred_probs, i_pred_probs, actual, know_prob, tau,
                                                            threshold, True)
    else:  #
        # uniform average
        final_preds = (k_pred_probs + i_pred_probs) / 2
        weights = None

    return final_preds, weights


def compute_switched_predictions(k_stat, i_stat, final_probs, q_pred_probs, num_signals, freq_type, freq_agg):
    assert len(final_probs) == len(q_pred_probs)
    new_probs = np.zeros(len(final_probs))

    for idx in range(len(final_probs)):
        is_valid = True
        num_vids = 0
        if freq_agg == "n_events":
            n_signals = idx
        elif freq_agg == "n_vid":
            if idx == 0:
                is_new_vid = True
            else:
                is_new_vid = bool(k_stat["video_change"][idx])
            is_valid = is_new_vid
            if is_valid:
                num_vids += 1
            n_signals = num_vids - 1
        else:
            skill_freqs = []
            if "k" in freq_type:
                skill_freqs += k_stat["num_updates"][idx]
            if "i" in freq_type:
                skill_freqs += i_stat["num_updates"][idx]

            if freq_agg == "sum":
                n_signals = sum(skill_freqs)
            elif freq_agg == "min":
                n_signals = min(skill_freqs)
            else:
                n_signals = sum([bool(x) for x in k_stat["num_updates"][idx]])

        if is_valid and (n_signals < num_signals):
            new_probs[idx] = q_pred_probs[idx]
        else:
            new_probs[idx] = final_probs[idx]

    return new_probs
    # final_probs[0] = q_pred_probs[0]
    # return final_probs


def hybrid_truelearn_model(records, init_skill=0., k_def_var=None, i_def_var=None, tau=0., k_beta_sqr=0., i_beta_sqr=0.,
                           threshold=0.5, draw_probability="static", draw_factor=.1, positive_only=False,
                           semantic_mapping=None, tracking=False, agg_func="raw", is_pred_only=False, is_diluted=False,
                           dil_factor=1.0, var_const=0., top_k_sr_topics=-1, sr_func="raw", is_topics=False,
                           decay_func=None, prob_combine_type="weight", know_prob=1., k_topics=1, i_topics=1,
                           start_event=0, q_random=False):
    num_records = float(len(records))

    if num_records <= 1:
        return 0., [], int(num_records), False

    k_user_model = {
        "mean": {},
        "variance": {},
        "updates": {},
        "last_time": {}
    }

    i_user_model = {
        "mean": {},
        "variance": {},
        "updates": {},
        "last_time": {}
    }

    k_topic_cov = set()
    i_topic_cov = set()

    k_stat = defaultdict(int)
    i_stat = defaultdict(int)

    if not (semantic_mapping is None or type(semantic_mapping) == dict):  # convert broadcast variable to dict
        semantic_mapping = semantic_mapping.value

    # get knowledge predictions
    actual, k_pred_probs, k_user_model, k_topic_cov, k_stat = predict_and_model_truelearn_novel(records, k_user_model,
                                                                                                draw_probability,
                                                                                                draw_factor,
                                                                                                k_topic_cov,
                                                                                                positive_only,
                                                                                                k_stat,
                                                                                                init_skill,
                                                                                                k_def_var, k_beta_sqr,
                                                                                                0.,
                                                                                                semantic_mapping,
                                                                                                agg_func,
                                                                                                is_pred_only,
                                                                                                is_diluted,
                                                                                                dil_factor,
                                                                                                var_const,
                                                                                                top_k_sr_topics,
                                                                                                sr_func, tracking,
                                                                                                is_topics,
                                                                                                False,  # is fixed
                                                                                                False,  # is interest
                                                                                                decay_func,  #
                                                                                                k_topics, threshold,
                                                                                                start_event,
                                                                                                q_random)

    # actual, i_pred_probs, i_user_model, i_topic_cov, i_stat = predict_and_model_kt_model(records, i_user_model,
    #                                                                                      i_def_var,
    #                                                                                      tau,  # pfail
    #                                                                                      i_beta_sqr,  # pguess
    #                                                                                      i_stat,
    #                                                                                      i_topic_cov, decay_func,
    #                                                                                      False,  # is interest
    #                                                                                      positive_only)

    # Gaussian Interest ===============
    actual, i_pred_probs, i_user_model, i_topic_cov, i_stat = predict_and_model_truelearn_novel(records, i_user_model,
                                                                                                draw_probability,
                                                                                                draw_factor,
                                                                                                i_topic_cov,
                                                                                                positive_only,
                                                                                                i_stat,
                                                                                                init_skill,
                                                                                                i_def_var, i_beta_sqr,
                                                                                                0.,
                                                                                                semantic_mapping,
                                                                                                agg_func,
                                                                                                is_pred_only,
                                                                                                is_diluted,
                                                                                                dil_factor,
                                                                                                var_const,
                                                                                                top_k_sr_topics,
                                                                                                sr_func, tracking,
                                                                                                is_topics,
                                                                                                True,  # is_fixed
                                                                                                True,  # is_interest
                                                                                                decay_func,
                                                                                                i_topics, threshold,
                                                                                                start_event,
                                                                                                q_random)

    k_stat["num_topics"] = len(k_topic_cov)
    i_stat["num_topics"] = len(i_topic_cov)

    k_pred_probs = np.array(k_pred_probs).astype("float64")
    i_pred_probs = np.array(i_pred_probs).astype("float64")
    actual = np.array(actual).astype("int")

    final_probs, weights = compute_hybrid_probabilities(k_pred_probs, i_pred_probs, prob_combine_type, actual,
                                                        know_prob, tau, threshold)

    final_probs = final_probs.astype("float64")

    accuracy, precision, recall, f1, roc_score, pr_score, stats = get_summary_stats(actual, final_probs, num_records,
                                                                                    stats=k_stat,
                                                                                    user_model=k_user_model,
                                                                                    threshold=threshold,
                                                                                    k_preds=k_pred_probs,
                                                                                    i_preds=i_pred_probs,
                                                                                    i_stats=i_stat,
                                                                                    i_user_model=i_user_model,
                                                                                    weights=weights)

    return accuracy, precision, recall, f1, roc_score, pr_score, int(num_records), stats


def truelearn_novelq_model(records, init_skill=0., k_def_var=None, i_def_var=None, tau=0., k_beta_sqr=0., i_beta_sqr=0.,
                           threshold=0.5, draw_probability="static", draw_factor=.1, positive_only=False,
                           semantic_mapping=None, tracking=False, agg_func="raw", is_pred_only=False, is_diluted=False,
                           dil_factor=1.0, var_const=0., top_k_sr_topics=-1, sr_func="raw", is_topics=False,
                           decay_func=None, prob_combine_type="weight", know_prob=1., k_topics=1, i_topics=1,
                           start_event=0, quality_mapping=None, num_signals=1, freq_type="k", quality_type="prediction",
                           freq_agg="sum", weighted=False):
    num_records = float(len(records))

    if num_records <= 1:
        return 0., [], int(num_records), False

    k_user_model = {
        "mean": {},
        "variance": {},
        "updates": {},
        "last_time": {}
    }

    k_topic_cov = set()

    k_stat = defaultdict(int)

    if not (semantic_mapping is None or type(semantic_mapping) == dict):  # convert broadcast variable to dict
        semantic_mapping = semantic_mapping.value

    # get knowledge predictions
    actual, k_pred_probs, k_user_model, k_topic_cov, k_stat = predict_and_model_truelearn_novel(records, k_user_model,
                                                                                                draw_probability,
                                                                                                draw_factor,
                                                                                                k_topic_cov,
                                                                                                positive_only,
                                                                                                k_stat,
                                                                                                init_skill,
                                                                                                k_def_var, k_beta_sqr,
                                                                                                0.,
                                                                                                semantic_mapping,
                                                                                                agg_func,
                                                                                                is_pred_only,
                                                                                                is_diluted,
                                                                                                dil_factor,
                                                                                                var_const,
                                                                                                top_k_sr_topics,
                                                                                                sr_func, tracking,
                                                                                                is_topics,
                                                                                                False,  # is fixed
                                                                                                False,  # is interest
                                                                                                decay_func,  #
                                                                                                k_topics, threshold,
                                                                                                start_event,
                                                                                                False)

    q_pred_probs = compute_quality_based_predictions(records, quality_mapping, start_event, 5)

    k_stat["num_topics"] = len(k_topic_cov)

    k_pred_probs = np.array(k_pred_probs).astype("float64")

    actual = np.array(actual).astype("int")

    final_probs = compute_switched_predictions(k_stat, {}, k_pred_probs, q_pred_probs, num_signals, freq_type,
                                               "n_events")

    final_probs = final_probs.astype("float64")

    accuracy, precision, recall, f1, roc_score, pr_score, stats = get_summary_stats(actual, final_probs, num_records,
                                                                                    stats=k_stat,
                                                                                    user_model=k_user_model,
                                                                                    threshold=threshold,
                                                                                    k_preds=k_pred_probs)

    return accuracy, precision, recall, f1, roc_score, pr_score, int(num_records), stats


def qink_truelearn_model(records, init_skill=0., k_def_var=None, i_def_var=None, tau=0., k_beta_sqr=0., i_beta_sqr=0.,
                         threshold=0.5, draw_probability="static", draw_factor=.1, positive_only=False,
                         semantic_mapping=None, tracking=False, agg_func="raw", is_pred_only=False, is_diluted=False,
                         dil_factor=1.0, var_const=0., top_k_sr_topics=-1, sr_func="raw", is_topics=False,
                         decay_func=None, prob_combine_type="weight", know_prob=1., k_topics=1, i_topics=1,
                         start_event=0, quality_mapping=None, quality_type="prediction", num_signals=1, freq_type="k",
                         freq_agg="sum", weighted=False):
    num_records = float(len(records))

    if num_records <= 1:
        return 0., [], int(num_records), False

    k_user_model = {
        "mean": {},
        "variance": {},
        "updates": {},
        "last_time": {}
    }

    i_user_model = {
        "mean": {},
        "variance": {},
        "updates": {},
        "last_time": {}
    }

    k_topic_cov = set()
    i_topic_cov = set()

    k_stat = defaultdict(int)
    i_stat = defaultdict(int)

    if not (semantic_mapping is None or type(semantic_mapping) == dict):  # convert broadcast variable to dict
        semantic_mapping = semantic_mapping.value

    # get knowledge predictions
    actual, k_pred_probs, k_user_model, k_topic_cov, k_stat = predict_and_model_truelearn_novel(records, k_user_model,
                                                                                                draw_probability,
                                                                                                draw_factor,
                                                                                                k_topic_cov,
                                                                                                positive_only,
                                                                                                k_stat,
                                                                                                init_skill,
                                                                                                k_def_var, k_beta_sqr,
                                                                                                0.,
                                                                                                semantic_mapping,
                                                                                                agg_func,
                                                                                                is_pred_only,
                                                                                                is_diluted,
                                                                                                dil_factor,
                                                                                                var_const,
                                                                                                top_k_sr_topics,
                                                                                                sr_func, tracking,
                                                                                                is_topics,
                                                                                                False,  # is fixed
                                                                                                False,  # is interest
                                                                                                decay_func,  #
                                                                                                k_topics, threshold,
                                                                                                start_event,
                                                                                                False)

    # Gaussian Interest ===============
    actual, i_pred_probs, i_user_model, i_topic_cov, i_stat = predict_and_model_truelearn_novel(records, i_user_model,
                                                                                                draw_probability,
                                                                                                draw_factor,
                                                                                                i_topic_cov,
                                                                                                positive_only,
                                                                                                i_stat,
                                                                                                init_skill,
                                                                                                i_def_var, i_beta_sqr,
                                                                                                0.,
                                                                                                semantic_mapping,
                                                                                                agg_func,
                                                                                                is_pred_only,
                                                                                                is_diluted,
                                                                                                dil_factor,
                                                                                                var_const,
                                                                                                top_k_sr_topics,
                                                                                                sr_func, tracking,
                                                                                                is_topics,
                                                                                                True,  # is_fixed
                                                                                                True,  # is_interest
                                                                                                decay_func,
                                                                                                i_topics, threshold,
                                                                                                start_event, False)

    q_pred_probs = compute_quality_based_predictions(records, quality_mapping, start_event, 5)

    k_stat["num_topics"] = len(k_topic_cov)
    i_stat["num_topics"] = len(i_topic_cov)

    k_pred_probs = np.array(k_pred_probs).astype("float64")
    i_pred_probs = np.array(i_pred_probs).astype("float64")
    actual = np.array(actual).astype("int")

    final_probs, weights = compute_hybrid_probabilities(k_pred_probs, i_pred_probs, prob_combine_type, actual,
                                                        know_prob, tau, threshold, weighted=weighted,
                                                        quality_preds=q_pred_probs)

    if not weighted:  # if not weighted, switch
        # add population prior
        final_probs = compute_switched_predictions(k_stat, i_stat, final_probs, q_pred_probs, num_signals, freq_type,
                                                   freq_agg)

    final_probs = final_probs.astype("float64")

    accuracy, precision, recall, f1, roc_score, pr_score, stats = get_summary_stats(actual, final_probs, num_records,
                                                                                    stats=k_stat,
                                                                                    user_model=k_user_model,
                                                                                    threshold=threshold,
                                                                                    k_preds=k_pred_probs,
                                                                                    i_preds=i_pred_probs,
                                                                                    i_stats=i_stat,
                                                                                    i_user_model=i_user_model,
                                                                                    weights=weights)

    return accuracy, precision, recall, f1, roc_score, pr_score, int(num_records), stats
