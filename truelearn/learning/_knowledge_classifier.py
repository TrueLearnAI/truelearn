from __future__ import annotations
from typing import Iterable, Hashable
import math

import trueskill

from truelearn.models import AbstractKnowledge, AbstractKnowledgeComponent, LearnerModel


class KnowledgeClassifier:
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
        Train the model based on the given data and label.
    predict(x)
        Predict whether the learner will engage.
    predict_proba(x)
        Predict the probability of learner engagement.
    get_params()
        Get parameters associated with the model.

    """

    CONTENT_VARIANCE = 1e-9

    def __init__(self, *, learner_model: LearnerModel | None = None, threshold: float = 0.5,
                 init_skill=0., def_var=0.5, beta: float = 0.5, positive_only=True) -> None:
        if learner_model is None:
            self.__learner_model = LearnerModel()
        else:
            self.__learner_model = learner_model
        self.__threshold = threshold
        self.__init_skill = init_skill
        self.__def_var = def_var
        self.__beta = beta
        self.__positive_only = positive_only

        # initialize an environment in which training will take place
        self.__env = trueskill.TrueSkill()
        trueskill.setup(mu=0., sigma=KnowledgeClassifier.CONTENT_VARIANCE, beta=float(self.__beta),
                        tau=float(self.__learner_model.tau), draw_probability=0.,
                        backend="mpmath", env=self.__env)

    def __topic_kc_pair_mapper(self, topic_kc_pair: tuple[Hashable, AbstractKnowledgeComponent]) -> tuple[Hashable, AbstractKnowledgeComponent]:
        """Retrieve a (topic_id, AbstractKnowledgeComponent) pair from learner model.

        If the AbstractKnowledge of the learner doesn't contain the topic_id,
        a new KC will be constructed via `kc.clone(self.__init_skill, self.__def_var)`.

        Parameters
        ----------
        topic_kc_pair : tuple[Hashable, AbstractKnowledgeComponent]
            The (topic_id, AbstractKnowledgeComponent) pair from the learnable unit.

        Returns
        -------
        tuple[Hashable, AbstractKnowledgeComponent]

        """
        topic_id, kc = topic_kc_pair
        extracted_kc = self.__learner_model.knowledge.get_kc(
            topic_id, kc.clone(self.__init_skill, self.__def_var))
        return topic_id, extracted_kc

    def __kc_mapper(self, topic_kc_pair: tuple[Hashable, AbstractKnowledgeComponent]) -> AbstractKnowledgeComponent:
        """Retrieve a KC from learner model.

        If the AbstractKnowledge of the learner doesn't contain the topic_id,
        a new KC will be constructed via `kc.clone(self.__init_skill, self.__def_var)`.

        Parameters
        ----------
        topic_kc_pair : tuple[Hashable, AbstractKnowledgeComponent]
            The (topic_id, AbstractKnowledgeComponent) pair from the learnable unit.

        Returns
        -------
        AbstractKnowledgeComponent

        """
        topic_id, kc = topic_kc_pair
        extracted_kc = self.__learner_model.knowledge.get_kc(
            topic_id, kc.clone(self.__init_skill, self.__def_var))
        return extracted_kc

    def __select_topic_kc_pairs(self, content_knowledge: AbstractKnowledge) -> Iterable[tuple[Hashable, AbstractKnowledgeComponent]]:
        """Return an iterable representing the learner's knowledge in the topics specified by the learnable unit.

        Given the knowledge representation of the learnable unit, this method tries to get
        the corresponding knowledge representation from the Learner Model.

        If it cannot find the corresponding knowledge component in learner's model, which means
        the learner has never exposed to this knowledge component before, a new KC will be constructed
        with initial skill and default variance. This will be done by using `__topic_kc_pair_mapper`.

        Parameters
        ----------
        content_knowledge : AbstractKnowledge
            The knowledge representation of a learnable unit.

        Returns
        -------
        Iterable[tuple[Hashable, AbstractKnowledgeComponent]]

        """
        team_learner = map(self.__topic_kc_pair_mapper,
                           content_knowledge.topic_kc_pairs())
        return team_learner

    def __select_kcs(self, content_knowledge: AbstractKnowledge) -> Iterable[AbstractKnowledgeComponent]:
        """Return an iterable of the KC representing the learner's knowledge in the topic specified by the learnable unit.

        Given the knowledge representation of the learnable unit, this method tries to get
        the corresponding knowledge representation from the Learner Model.

        If it cannot find the corresponding knowledge component in learner's model, which means
        the learner has never exposed to this knowledge component before, a new KC will be constructed
        with initial skill and default variance. This will be done by using `__kc_mapper`.

        Parameters
        ----------
        content_knowledge : AbstractKnowledge
            The knowledge representation of a learnable unit.

        Returns
        -------
        Iterable[AbstractKnowledgeComponent]

        """
        team_learner = map(self.__kc_mapper,
                           content_knowledge.topic_kc_pairs())
        return team_learner

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
                        sum(team_content_variance) + self.__beta)
        return float(self.__env.
                     cdf(difference, 0, std))  # type: ignore

    def fit(self, x: AbstractKnowledge, y: bool) -> KnowledgeClassifier:
        """Train the model based on a given AbstractKnowledge that represents a learnable unit.

        Parameters
        ----------
        x : AbstractKnowledge
            A knowledge representation of a learnable unit.
        y : bool
            Whether the learner engages with the learnable unit.

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
        if self.__positive_only and x is False:
            return self

        learner_topic_kc_pairs = list(self.__select_topic_kc_pairs(x))
        content_kcs = x.knowledge_components()

        team_learner = tuple(
            map(
                lambda topic_kc_learner_pair: self.__env.create_rating(
                    mu=topic_kc_learner_pair[1].mean, sigma=math.sqrt(topic_kc_learner_pair[1].variance)),
                learner_topic_kc_pairs
            )
        )
        team_content = tuple(
            map(
                lambda kc: self.__env.create_rating(
                    mu=kc.mean, sigma=math.sqrt(kc.variance)),
                content_kcs
            )
        )

        if y:
            # learner wins: lower rank == winning
            updated_team_learner, _ = self.__env.rate(
                [team_learner, team_content], ranks=[0, 1])
        else:
            # content wins
            _, updated_team_learner = self.__env.rate(
                [team_content, team_learner], ranks=[0, 1])

        for topic_kc_pair, rating in zip(learner_topic_kc_pairs, updated_team_learner):
            topic_id, kc = topic_kc_pair
            kc.update(rating.mean, rating.sigma ** 2)
            self.__learner_model.knowledge.update_kc(topic_id, kc)

        return self

    def predict(self, x: AbstractKnowledge) -> bool:
        """Predict whether the learner will engage with the given learnable unit.

        The function will return True iff the probability that the learner engages
        with the learnable unit is greater than the given threshold.

        Refer to `predict_proba` for more details.

        Parameters
        ----------
        x : AbstractKnowledge
            A knowledge representation of a learnable unit.

        Returns
        -------
        bool
            Whether the learner will engage with the given learnable unit.

        """
        return self.predict_proba(x) > self.__threshold

    def predict_proba(self, x: AbstractKnowledge) -> float:
        """Predict the probability of the learner's engagement with the given learnable unit.

        Learner and Learnable Unit is can be represented as a Normal Distribution with certain skills (mu) and
        standard deviation (sqrt{variance}).

        The algorithm uses cumulative density function of normal distribution to calculate the probability.
        It calculates the probability of getting x in a Normal Distribution N(0, std) where x is the difference
        between the learner's skill (mean) and the learnable unit's skill (mean) and std is the standard deviation
        of the new normal distribution as a result of subtracting the two old normal distribution (learner and
        learnable unit).

        Parameters
        ----------
        x : AbstractKnowledge
            A knowledge representation of a learnable unit.

        Returns
        -------
        float
            The probability that the learner engages with the given learnable unit.

        """
        learner_kcs = self.__select_kcs(x)
        content_kcs = x.knowledge_components()
        return self.__team_sum_quality(learner_kcs, content_kcs)

    def get_params(self) -> dict[str, float]:
        """Get the parameters associated with the model.

        Returns
        -------
        dict[str, float]
            A parameter name and value pair.

        """
        return {
            "threshold": self.__threshold,
            "init_skill": self.__init_skill,
            "def_var": self.__def_var,
            "beta": self.__beta,
            "positive_only": self.__positive_only
        }
