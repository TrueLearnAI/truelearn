from typing import Iterable

from sklearn import metrics


def get_precision_score(
    pred_labels: Iterable[bool], act_labels: Iterable[bool]
) -> float:
    """Get the precision score of the prediction.

    Args:
        pred_labels: An iterable of predicted labels
        act_labels: An iterable of actual labels

    Returns:
        The precision score.
    """
    return float(metrics.precision_score(pred_labels, act_labels))


def get_recall_score(pred_labels: Iterable[bool], act_labels: Iterable[bool]) -> float:
    """Get the recall score of the prediction.

    Args:
        pred_labels: An iterable of predicted labels
        act_labels: An iterable of actual labels

    Returns:
        The recall score.
    """
    return float(metrics.recall_score(pred_labels, act_labels))


def get_accuracy_score(
    pred_labels: Iterable[bool], act_labels: Iterable[bool]
) -> float:
    """Get the accuracy score of the prediction.

    Args:
        pred_labels: An iterable of predicted labels
        act_labels: An iterable of actual labels

    Returns:
        The accuracy score.
    """
    return float(metrics.accuracy_score(pred_labels, act_labels))


def get_f1_score(pred_labels: Iterable[bool], act_labels: Iterable[bool]) -> float:
    """Get the f1 score of the prediction.

    Args:
        pred_labels: An iterable of predicted labels
        act_labels: An iterable of actual labels

    Returns:
        The f1 score.
    """
    return float(metrics.f1_score(pred_labels, act_labels))
