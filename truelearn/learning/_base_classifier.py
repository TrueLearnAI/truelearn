from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, Hashable

import trueskill

from truelearn.models import EventModel, AbstractKnowledge, AbstractKnowledgeComponent, LearnerModel


class BaseClassifier(ABC):
    """The base class of all the classifiers in TrueLearn.

    Notes
    -----
    This classifier checks the type of the parameters.
    So, all estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments.

    Inspired by scikit-learn BaseEstimator.

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

    # TODO: fix draw_probability type
    def __init__(self, *, learner_model: LearnerModel | None = None, threshold: float = 0.5,
                 init_skill=0., def_var=0.5, beta: float = 0.5, positive_only=True, draw_probability: float = 0.1) -> None:
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
        self._draw_probability = draw_probability

        # initialize an environment in which training will take place
        self._env = trueskill.TrueSkill()
        trueskill.setup(mu=0., sigma=InterestNoveltyKnowledgeBaseClassifier.CONTENT_VARIANCE, beta=float(self._beta),
                        tau=float(self._learner_model.tau), draw_probability=self._draw_probability,
                        backend="mpmath", env=self._env)

    def _topic_kc_pair_mapper(self, topic_kc_pair: tuple[Hashable, AbstractKnowledgeComponent]) -> tuple[Hashable, AbstractKnowledgeComponent]:
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
        extracted_kc = self._learner_model.knowledge.get_kc(
            topic_id, kc.clone(self._init_skill, self._def_var))
        return topic_id, extracted_kc

    def _kc_mapper(self, topic_kc_pair: tuple[Hashable, AbstractKnowledgeComponent]) -> AbstractKnowledgeComponent:
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
        extracted_kc = self._learner_model.knowledge.get_kc(
            topic_id, kc.clone(self._init_skill, self._def_var))
        return extracted_kc

    def _select_topic_kc_pairs(self, content_knowledge: AbstractKnowledge) -> Iterable[tuple[Hashable, AbstractKnowledgeComponent]]:
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
        team_learner = map(self._topic_kc_pair_mapper,
                           content_knowledge.topic_kc_pairs())
        return team_learner

    def _select_kcs(self, content_knowledge: AbstractKnowledge) -> Iterable[AbstractKnowledgeComponent]:
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
        team_learner = map(self._kc_mapper,
                           content_knowledge.topic_kc_pairs())
        return team_learner
