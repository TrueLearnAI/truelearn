from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, Hashable
from statistics import NormalDist
from math import sqrt

import trueskill

from truelearn.models import EventModel, AbstractKnowledge, AbstractKnowledgeComponent, LearnerModel


class BaseClassifier(ABC):
    """The base class of all the classifiers in TrueLearn.

    # TODO: add type checking

    Notes
    -----
    This classifier checks the type of the parameters.
    So, all estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments.

    """

    @abstractmethod
    def fit(self, x: EventModel, y: bool) -> BaseClassifier:
        """Train the model based on a given EventModel that represents a learning event.

        Parameters
        ----------
        x : EventModel
            A representation of a learning event.
        y : bool
            Whether the learner engages in the given learning event.

        Returns
        -------
        BaseClassifier
            The updated BaseClassifier.

        """

    @abstractmethod
    def predict(self, x: EventModel) -> bool:
        """Predict whether the learner will engage in the given learning event.

        Parameters
        ----------
        x : EventModel
            A representation of a learning event.

        Returns
        -------
        bool
            Whether the learner will engage in the given learning event.

        """

    @abstractmethod
    def predict_proba(self, x: EventModel) -> float:
        """Predict the probability of the learner's engagement in the given learning event.

        Parameters
        ----------
        x : EventModel
            A representation of a learning event.

        Returns
        -------
        float
            The probability that the learner engages in the given learning event.

        """

    def get_params(self) -> dict:
        """Get the parameters associated with the classifier.

        Returns
        -------
        dict[str, float]
            A parameter name and value pair.

        """
        raise NotImplementedError("get_params is not implemented yet")

    def set_params(self, **params) -> BaseClassifier:
        """Set the parameters associated with the classifier.

        Returns
        -------
        BaseClassifier
            The updated BaseClassifier.

        """
        raise NotImplementedError("set_params is not implemented yet")


class InterestNoveltyKnowledgeBaseClassifier(BaseClassifier):
    """A Base Classifier for KnowledgeClassifier, NoveltyClassifier and InterestClassifier.

    It defines the necessary instance variables and
    common methods to interact with the LearnerModel.

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

    # TODO: fix probability doc

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

    DEFAULT_CONTENT_VARIANCE: float = 1e-9
    DEFAULT_DRAW_PROBA_LOW: float = 1e-9
    DEFAULT_DRAW_PROBA_HIGH: float = 0.999999999

    def __init__(self, *, learner_model: LearnerModel | None, threshold: float,
                 init_skill, def_var, beta: float, positive_only, draw_proba_type: str,
                 draw_proba_static: float, draw_proba_factor: float) -> None:
        super().__init__()

        if learner_model is None:
            self._learner_model = LearnerModel()
        else:
            self._learner_model = learner_model
        self._threshold = threshold
        self._init_skill = init_skill
        self._def_var = def_var
        self._beta = beta
        self._positive_only = positive_only

        self._draw_proba_type = draw_proba_type
        self._draw_proba_static = draw_proba_static
        self._draw_proba_factor = draw_proba_factor

        if self._draw_proba_type == "static":
            def __calculate_draw_proba_static():
                return self._draw_proba_static * self._draw_proba_factor
            self.__calculate_draw_proba = __calculate_draw_proba_static
        elif self._draw_proba_type == "dynamic":
            def __calculate_draw_proba_dynamic():
                total_engagement_stats = min(
                    1, self._learner_model.number_of_engagements + self._learner_model.number_of_non_engagements)
                draw_probability = float(
                    self._learner_model.number_of_engagements / total_engagement_stats)

                # clamp the value between [DEFAULT_DRAW_PROBA_LOW, DEFAULT_DRAW_PROBA_HIGH]
                draw_probability = max(min(InterestNoveltyKnowledgeBaseClassifier.DEFAULT_DRAW_PROBA_HIGH,
                                       draw_probability), InterestNoveltyKnowledgeBaseClassifier.DEFAULT_DRAW_PROBA_LOW)

                # draw_proba_param is a factor if the type is dynamic
                return draw_probability * self._draw_proba_factor

            self.__calculate_draw_proba = __calculate_draw_proba_dynamic
        else:
            raise RuntimeError(
                "Unsupported draw probability type. The model supports two types: static and dynamic")

        self._draw_probability = self.__calculate_draw_proba()

        # create an environment in which training will take place
        self._env = trueskill.TrueSkill()

    def _gather_trueskill_team(self, kcs: Iterable[AbstractKnowledgeComponent]) -> tuple[trueskill.Rating]:
        return tuple(map(
            lambda kc: self._env.create_rating(
                mu=kc.mean, sigma=sqrt(kc.variance)),
            kcs
        ))

    def __setup_env(self) -> None:
        self.__calculate_draw_proba()
        trueskill.setup(mu=0., sigma=InterestNoveltyKnowledgeBaseClassifier.DEFAULT_CONTENT_VARIANCE, beta=float(self._beta),
                        tau=float(self._learner_model.tau), draw_probability=self._draw_probability,
                        backend="mpmath", env=self._env)

    def __update_engagement_stats(self, y) -> None:
        if y:
            self._learner_model.number_of_engagements += 1
        else:
            self._learner_model.number_of_non_engagements += 1

    @abstractmethod
    def _update_knowledge_representation(self, x, y) -> None:
        """Update the knowledge representation of the LearnerModel.

        Parameters
        ----------
        x : EventModel
            A representation of a learning event.
        y : bool
            Whether the learner engages in the given learning event.

        """

    def fit(self, x: EventModel, y: bool) -> InterestNoveltyKnowledgeBaseClassifier:
        """Train the model based on a given EventModel that represents a learning event.

        Parameters
        ----------
        x : EventModel
            A representation of a learning event.
        y : bool
            Whether the learner engages in the given learning event.

        Returns
        -------
        InterestNoveltyKnowledgeBaseClassifier
            The updated InterestNoveltyKnowledgeBaseClassifier.

        """
        # update the knowledge representation if positive_only is False or (it's true and y is true)
        if not self._positive_only or y is True:
            self.__setup_env()
            self._update_knowledge_representation(x, y)

        self.__update_engagement_stats(y)
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


def team_sum_quality(learner_kcs: Iterable[AbstractKnowledgeComponent], content_kcs: Iterable[AbstractKnowledgeComponent], beta: float) -> float:
    """Return the probability that the learner engages with the learnable unit.

    Parameters
    ----------
    learner_kcs : Iterable[AbstractKnowledgeComponent]
        An iterable of learner's knowledge component.
    content_kcs : Iterable[AbstractKnowledgeComponent]
        An iterable of learnable unit's knowledge component.
    beta : float
        The noise factor, which is used in trueskill.

    Returns
    -------
    float

    """
    team_learner_mean = map(lambda kc: kc.mean, learner_kcs)
    team_learner_variance = map(lambda kc: kc.variance, learner_kcs)
    team_content_mean = map(lambda kc: kc.mean, content_kcs)
    team_content_variance = map(lambda kc: kc.variance, content_kcs)

    difference = sum(team_learner_mean) - sum(team_content_mean)
    std = sqrt(sum(team_learner_variance) + sum(team_content_variance) + beta)
    return NormalDist(mu=0, sigma=std).cdf(difference)


def select_topic_kc_pairs(learner_model: LearnerModel, content_knowledge: AbstractKnowledge,
                          init_skill: float, def_var: float) -> Iterable[tuple[Hashable, AbstractKnowledgeComponent]]:
    """Return an iterable representing the learner's knowledge in the topics specified by the learnable unit.

    Given the knowledge representation of the learnable unit, this method tries to get
    the corresponding knowledge representation from the Learner Model.

    If it cannot find the corresponding knowledge component in learner's model, which means
    the learner has never exposed to this knowledge component before, a new KC will be constructed
    with initial skill and default variance. This will be done by using `__topic_kc_pair_mapper`.

    Parameters
    ----------
    learner_model : LearnerModel
        A representation of the learner.
    content_knowledge : AbstractKnowledge
        The knowledge representation of a learnable unit.
    init_skill: float
        The initial skill (mean) of the learner given a new AbstractKnowledgeComponent.
    def_var: float
        The default variance of the new AbstractKnowledgeComponent.

    Returns
    -------
    Iterable[tuple[Hashable, AbstractKnowledgeComponent]]

    """
    def __topic_kc_pair_mapper(topic_kc_pair: tuple[Hashable, AbstractKnowledgeComponent]) -> tuple[Hashable, AbstractKnowledgeComponent]:
        topic_id, kc = topic_kc_pair
        extracted_kc = learner_model.knowledge.get_kc(
            topic_id, kc.clone(init_skill, def_var))
        return topic_id, extracted_kc

    team_learner = map(__topic_kc_pair_mapper,
                       content_knowledge.topic_kc_pairs())
    return team_learner


def select_kcs(learner_model: LearnerModel, content_knowledge: AbstractKnowledge,
               init_skill: float, def_var: float) -> Iterable[AbstractKnowledgeComponent]:
    """Return an iterable representing the learner's knowledge in the topics specified by the learnable unit.

    Given the knowledge representation of the learnable unit, this method tries to get
    the corresponding knowledge representation from the Learner Model.

    If it cannot find the corresponding knowledge component in learner's model, which means
    the learner has never exposed to this knowledge component before, a new KC will be constructed
    with initial skill and default variance. This will be done by using `__topic_kc_pair_mapper`.

    Parameters
    ----------
    learner_model : LearnerModel
        A representation of the learner.
    content_knowledge : AbstractKnowledge
        The knowledge representation of a learnable unit.
    init_skill: float
        The initial skill (mean) of the learner given a new AbstractKnowledgeComponent.
    def_var: float
        The default variance of the new AbstractKnowledgeComponent.

    Returns
    -------
    Iterable[tuple[Hashable, AbstractKnowledgeComponent]]

    """
    def __kc_mapper(topic_kc_pair: tuple[Hashable, AbstractKnowledgeComponent]) -> AbstractKnowledgeComponent:
        topic_id, kc = topic_kc_pair
        extracted_kc = learner_model.knowledge.get_kc(
            topic_id, kc.clone(init_skill, def_var))
        return extracted_kc

    team_learner = map(__kc_mapper, content_knowledge.topic_kc_pairs())
    return team_learner
