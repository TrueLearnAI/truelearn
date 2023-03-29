from typing import Iterable, Optional

from sklearn import metrics


def get_precision_score(
    act_labels: Iterable[bool],
    pred_labels: Iterable[bool],
    zero_division: Optional[int] = None,
) -> float:
    """Get the precision score of the prediction.

    The precision is the ratio `tp / (tp + fp)` where `tp` is the number of true
    positives and `fp` the number of false positives.
    The precision is intuitively the ability of the classifier not to label as positive
    a sample that is negative.

    Args:
        act_labels:
            An iterable of actual labels
        pred_labels:
            An iterable of predicted labels
        zero_division:
            Sets the value to return when there is a zero division.
            Defaults to None, which sets the value to zero and raises a warning.
            Acceptable values are 0 or 1, which sets the resulting value
            according.

    Returns:
        The precision score.
    """
    # the value check (whether it's 0 or 1) is done inside scikit-learn
    if zero_division is None:
        zero_division = "warn"  # type: ignore

    return float(
        metrics.precision_score(
            act_labels, pred_labels, zero_division=zero_division  # type: ignore
        )
    )


def get_recall_score(
    act_labels: Iterable[bool],
    pred_labels: Iterable[bool],
    zero_division: Optional[int] = None,
) -> float:
    """Get the recall score of the prediction.

    The recall is the ratio `tp / (tp + fn)` where `tp` is the number of true positives
    and `fn` the number of false negatives.
    The recall is intuitively the ability of the classifier to find all the
    positive samples.

    Args:
        act_labels:
            An iterable of actual labels
        pred_labels:
            An iterable of predicted labels
        zero_division:
            Sets the value to return when there is a zero division.
            Defaults to None, which sets the value to zero and raises a warning.
            Acceptable values are 0 or 1, which sets the resulting value
            according.

    Returns:
        The recall score.
    """
    # the value check (whether it's 0 or 1) is done inside scikit-learn
    if zero_division is None:
        zero_division = "warn"  # type: ignore

    return float(
        metrics.recall_score(
            act_labels, pred_labels, zero_division=zero_division  # type: ignore
        )
    )


def get_accuracy_score(
    act_labels: Iterable[bool], pred_labels: Iterable[bool]
) -> float:
    """Get the accuracy score of the prediction.

    The precision is the ratio `(tp + tn) / (tp + fp + tn + fn)` where `tp` is the
    number of true positives, `tn` is the number of true negatives,
    `fp` the number of false positives and `fn` is the number of false negatives.
    The accuracy is intuitively the ability of the classifier to correctly classify
    samples.

    Args:
        act_labels: An iterable of actual labels
        pred_labels: An iterable of predicted labels

    Returns:
        The accuracy score.
    """
    return float(metrics.accuracy_score(act_labels, pred_labels))


def get_f1_score(
    act_labels: Iterable[bool],
    pred_labels: Iterable[bool],
    zero_division: Optional[int] = None,
) -> float:
    """Get the f1 score of the prediction.

    The formula for the F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    The highest possible value of an F-score is 1.0, indicating perfect precision and
    recall, and the lowest possible value is 0, if either precision or recall are zero.

    Args:
        act_labels:
            An iterable of actual labels
        pred_labels:
            An iterable of predicted labels
        zero_division:
            Sets the value to return when there is a zero division.
            Defaults to None, which sets the value to zero and raises a warning.
            Acceptable values are 0 or 1, which sets the resulting value
            according.

    Returns:
        The f1 score.
    """
    # the value check (whether it's 0 or 1) is done inside scikit-learn
    if zero_division is None:
        zero_division = "warn"  # type: ignore

    return float(
        metrics.f1_score(
            act_labels, pred_labels, zero_division=zero_division  # type: ignore
        )
    )
