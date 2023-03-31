"""The truelearn.utils.metrics implements the commonly-used metrics in TrueLearn."""

from ._classification import (
    get_precision_score,
    get_recall_score,
    get_accuracy_score,
    get_f1_score,
)

__all__ = [
    "get_precision_score",
    "get_recall_score",
    "get_accuracy_score",
    "get_f1_score",
]
