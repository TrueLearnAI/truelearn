from typing import Iterable
import dataclasses

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


@dataclasses.dataclass
class ConfusionMatrix:
    """Confusion Matrix.

    Args:
        precision_score: The precision score.
        recall_score: The recall score.
        accuracy_score: The accuracy score.
        f1_score: The f1 score.
    """

    precision_score: float
    recall_score: float
    accuracy_score: float
    f1_score: float


def get_confusion_matrix(
    pred_labels: Iterable[bool], act_labels: Iterable[bool]
) -> ConfusionMatrix:
    """Get the confusion matrix of the prediction.

    Args:
        pred_labels: An iterable of predicted labels.
        act_labels: An iterable of actual labels.

    Returns:
        The confusion matrix of the prediction.
    """
    # as we don't know if the iterable is lazy-evaluated
    # to avoid that it's an expensive operation
    # we construct a list based on them
    pred_labels = list(pred_labels)
    act_labels = list(act_labels)

    return ConfusionMatrix(
        precision_score=get_accuracy_score(pred_labels, act_labels=act_labels),
        recall_score=get_recall_score(pred_labels=pred_labels, act_labels=act_labels),
        accuracy_score=get_accuracy_score(
            pred_labels=pred_labels, act_labels=act_labels
        ),
        f1_score=get_f1_score(pred_labels=pred_labels, act_labels=act_labels),
    )
