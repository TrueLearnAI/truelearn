"""The truelearn.learning module implements the classifiers in TrueLearn paper."""

from ._base import (
    InterestNoveltyKnowledgeBaseClassifier,
    draw_proba_static_constraint,
    team_sum_quality,
    team_sum_quality_from_kcs,
    gather_trueskill_team
)

from ._ink_classifier import INKClassifier
from ._interest_classifier import InterestClassifier
from ._knowledge_classifier import KnowledgeClassifier
from ._novelty_classifier import NoveltyClassifier
from ._engage_classifier import EngageClassifier
from ._persistent_classifier import PersistentClassifier
from ._majority_classifier import MajorityClassifier

__all__ = [
    "INKClassifier",
    "InterestClassifier",
    "InterestNoveltyKnowledgeBaseClassifier",
    "KnowledgeClassifier",
    "NoveltyClassifier",
    "EngageClassifier",
    "PersistentClassifier",
    "MajorityClassifier",
    "draw_proba_static_constraint",
    "team_sum_quality",
    "team_sum_quality_from_kcs"
]
