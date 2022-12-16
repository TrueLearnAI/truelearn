import numpy as np

from analyses.truelearn_experiments.utils import get_summary_stats


def engage_model(records):
    """
    predicts true to every instance
    Args:
        records:

    Returns:

    """
    num_records = len(records)

    if num_records <= 1:
        return 0., 0., 0., num_records, False

    predicted = np.ones(num_records)
    actual = [i[-1] for i in records]

    accuracy, precision, recall, f1, _ = get_summary_stats(actual, predicted, num_records)

    return accuracy, precision, recall, f1, num_records, False


def persistent_model(records, start_event=0):
    """This model always preserves the previous version of itself when it is modified.
    Args:
        records [[val]]: list of vectors for each event of the user. Format of vector
            [session, time, timeframe_id, topic_id, topic_cov ..., label]
        start_event (int): the index from which performance has to be recorded
    Returns:
        accuracy (float): accuracy for all observations
        concordance ([bool]): the concordance between actual and predicted values
    """

    num_records = len(records)

    if num_records <= 1:
        return 0., 0., 0., num_records, False

    prev_result = None

    topics_covered = set()

    actual = []
    predicted = []

    for idx, event in enumerate(records):
        label = event[-1]
        if idx >= start_event:
            prediction = prev_result

            actual.append(label)
            predicted.append(prediction)

        prev_result = label

    accuracy, precision, recall, f1, _ = get_summary_stats(actual, predicted, num_records)

    return accuracy, precision, recall, f1, num_records, False


def majority_model(records,start_event=0):
    """This model predicts the majority of the past as its current prediction.
        Args:
            records [[val]]: list of vectors for each event of the user. Format of vector
                [session, time, timeframe_id, topic_id, topic_cov ..., label]
            start_event (int): the index from which performance has to be recorded

        Returns:
            accuracy (float): accuracy for all observations
            concordance ([bool]): the concordance between actual and predicted values
    """

    num_records = len(records)

    if num_records <= 1:
        return 0., 0., 0., num_records, False

    actual = []
    predicted = []

    for idx, event in enumerate(records):
        if idx >= start_event:
            label = event[-1]
            prediction = int(np.mean([l[-1] for l in records[:idx]]) >= 0.5)
            actual.append(label)
            predicted.append(prediction)

    accuracy, precision, recall, f1, _ = get_summary_stats(actual, predicted, num_records)

    return accuracy, precision, recall, f1, num_records, False
