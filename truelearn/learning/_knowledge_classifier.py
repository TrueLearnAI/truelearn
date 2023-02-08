from __future__ import annotations
from typing import Iterable
import math

from ._base_classifier import InterestNoveltyKnowledgeBaseClassifier
from truelearn.models import EventModel, AbstractKnowledgeComponent, LearnerModel


class KnowledgeClassifier(InterestNoveltyKnowledgeBaseClassifier):
    """A Knowledge Classifier.

    Parameters
    ----------
    learner_model: LearnerModel | None, optional
    threshold: float
        Threshold for judging learner engagement. If the probability of the learner engagement is greater
        than the threshold, the model will predict engagement.
    init_skill: float
        The initial skill (mean) of the learner given a new KnowledgeComponent.
    def_var: float
        The default variance of the new KnowledgeComponent.
    beta: float
        The noise factor, which is used in trueskill.
    positive_only: bool
        Whether the model updates itself only if encountering positive data.

    Methods
    -------
    fit(x, y)
        Train the model based on the given event and label.
    predict(x)
        Predict whether the learner will engage.
    predict_proba(x)
        Predict the probability of learner engagement.
    get_params()
        Get parameters associated with the model.

    """

    def __init__(self, *, learner_model: LearnerModel | None = None, threshold: float = 0.5,
                 init_skill=0., def_var=0.5, beta: float = 0.5, positive_only=True) -> None:
        super().__init__(learner_model=learner_model, threshold=threshold,
                         init_skill=init_skill, def_var=def_var, beta=beta, positive_only=positive_only)

    def __team_sum_quality(self, learner_kcs: Iterable[AbstractKnowledgeComponent],
                           content_kcs: Iterable[AbstractKnowledgeComponent]) -> float:
        """Return the probability that the learner engages with the learnable unit.

        Parameters
        ----------
        learner_kcs : Iterable[AbstractKnowledgeComponent]
            An iterable of learner's knowledge component.
        content_kcs : Iterable[AbstractKnowledgeComponent]
            An iterable of learnable unit's knowledge component.

        Returns
        -------
        float

        """
        team_learner_mean = map(lambda kc: kc.mean, learner_kcs)
        team_learner_variance = map(lambda kc: kc.variance, learner_kcs)
        team_content_mean = map(lambda kc: kc.mean, content_kcs)
        team_content_variance = map(lambda kc: kc.variance, content_kcs)

        difference = sum(team_learner_mean) - sum(team_content_mean)
        std = math.sqrt(sum(team_learner_variance) +
                        sum(team_content_variance) + self._beta)
        return float(self._env.
                     cdf(difference, 0, std))  # type: ignore

    def _update_knowledge_representation(self, x: EventModel, y: bool) -> None:
        learner_topic_kc_pairs = list(self._select_topic_kc_pairs(x.knowledge))
        learner_kcs = map(
            lambda topic_kc_pair: topic_kc_pair[1], learner_topic_kc_pairs)
        content_kcs = x.knowledge.knowledge_components()

        team_learner = self._gather_trueskill_team(learner_kcs)
        team_content = self._gather_trueskill_team(content_kcs)

        if y:
            # learner wins: lower rank == winning
            updated_team_learner, _ = self._env.rate(
                [team_learner, team_content], ranks=[0, 1])
        else:
            # content wins
            _, updated_team_learner = self._env.rate(
                [team_content, team_learner], ranks=[0, 1])

        for topic_kc_pair, rating in zip(learner_topic_kc_pairs, updated_team_learner):
            topic_id, kc = topic_kc_pair
            kc.update(rating.mean, rating.sigma ** 2)
            self._learner_model.knowledge.update_kc(topic_id, kc)

    def predict_proba(self, x: EventModel) -> float:
        """Predict the probability of the learner's engagement in the given learning event.

        Learner and Learnable Unit is can be represented as a Normal Distribution with certain skills (mu) and
        standard deviation (sqrt{variance}).

        The algorithm uses cumulative density function of normal distribution to calculate the probability.
        It calculates the probability of getting x in a Normal Distribution N(0, std) where x is the difference
        between the learner's skill (mean) and the learnable unit's skill (mean) and std is the standard deviation
        of the new normal distribution as a result of subtracting the two old normal distribution (learner and
        learnable unit).

        # TODO: describe the win probability

        Parameters
        ----------
        x : EventModel
            A representation of a learning event.

        Returns
        -------
        float
            The probability that the learner engages in the given learning event.

        """
        learner_kcs = map(
            lambda x: x[1], self._select_topic_kc_pairs(x.knowledge))
        content_kcs = x.knowledge.knowledge_components()
        return self.__team_sum_quality(learner_kcs, content_kcs)
