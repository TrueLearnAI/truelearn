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

    def fit(self, x: EventModel, y: bool) -> KnowledgeClassifier:
        """Train the model based on a given EventModel that represents a learning event.

        Parameters
        ----------
        x : EventModel
            A representation of a learning event.
        y : bool
            Whether the learner engages in the given learning event.

        Returns
        -------
        KnowledgeClassifier
            The updated KnowledgeClassifier.

        References
        ----------
        [1] Bulathwela, S. et al. (2020) “TrueLearn: A Family of Bayesian algorithms to match lifelong learners
        to open educational resources,” Proceedings of the AAAI Conference on Artificial Intelligence, 34(01),
        pp. 565-573. Available at: https://doi.org/10.1609/aaai.v34i01.5395.

        """
        if self._positive_only and x is False:
            return self

        knowledge = x.knowledge
        learner_topic_kc_pairs = list(self._select_topic_kc_pairs(knowledge))
        content_kcs = knowledge.knowledge_components()

        team_learner = tuple(
            map(
                lambda topic_kc_learner_pair: self._env.create_rating(
                    mu=topic_kc_learner_pair[1].mean, sigma=math.sqrt(topic_kc_learner_pair[1].variance)),
                learner_topic_kc_pairs
            )
        )
        team_content = tuple(
            map(
                lambda content_kc: self._env.create_rating(
                    mu=content_kc.mean, sigma=math.sqrt(content_kc.variance)),
                content_kcs
            )
        )

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

        return self

    def predict(self, x: EventModel) -> bool:
        """Predict whether the learner will engage in the given learning event.

        The function will return True iff the probability that the learner engages
        with the learnable unit is greater than the given threshold.

        Refer to `predict_proba` for more details.

        Parameters
        ----------
        x : EventModel
            A representation of a learning event.

        Returns
        -------
        bool
            Whether the learner will engage in the given learning event.

        """
        return self.predict_proba(x) > self._threshold

    def predict_proba(self, x: EventModel) -> float:
        """Predict the probability of the learner's engagement in the given learning event.

        Learner and Learnable Unit is can be represented as a Normal Distribution with certain skills (mu) and
        standard deviation (sqrt{variance}).

        The algorithm uses cumulative density function of normal distribution to calculate the probability.
        It calculates the probability of getting x in a Normal Distribution N(0, std) where x is the difference
        between the learner's skill (mean) and the learnable unit's skill (mean) and std is the standard deviation
        of the new normal distribution as a result of subtracting the two old normal distribution (learner and
        learnable unit).

        Parameters
        ----------
        x : EventModel
            A representation of a learning event.

        Returns
        -------
        float
            The probability that the learner engages in the given learning event.

        """
        knowledge = x.knowledge
        learner_kcs = self._select_kcs(knowledge)
        content_kcs = knowledge.knowledge_components()
        return self.__team_sum_quality(learner_kcs, content_kcs)
