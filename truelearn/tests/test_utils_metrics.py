# pylint: disable=missing-function-docstring
import math

from truelearn.utils import metrics


def test_precision_score():
    y_true = [False, True, False, False, True, True]
    y_pred = [False, False, True, False, False, True]
    assert metrics.get_precision_score(y_true, y_pred) == 0.5


def test_recall_score():
    y_true = [False, True, False, False, True, True]
    y_pred = [False, False, True, False, False, True]
    assert math.isclose(metrics.get_recall_score(y_true, y_pred), 1 / 3)


def test_accuracy_score():
    y_true = [False, True, False, False, True, True]
    y_pred = [False, False, True, False, False, True]
    assert metrics.get_accuracy_score(y_true, y_pred) == 0.5


def test_f1_score():
    y_true = [False, True, False, False, True, True]
    y_pred = [False, False, True, False, False, True]
    assert metrics.get_f1_score(y_true, y_pred) == 0.4
