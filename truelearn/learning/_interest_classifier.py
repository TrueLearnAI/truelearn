from typing import Callable
from datetime import datetime as dt
import math

from ._base import InterestNoveltyKnowledgeBaseClassifier, team_sum_quality, select_kcs, select_topic_kc_pairs
from truelearn.models import EventModel, LearnerModel, AbstractKnowledgeComponent


class InterestClassifier(InterestNoveltyKnowledgeBaseClassifier):
    """An Interest Classifier.

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
        Train the model based on the given event and label.
    predict(x)
        Predict whether the learner will engage.
    predict_proba(x)
        Predict the probability of learner engagement.
    get_params()
        Get parameters associated with the model.

    """

    _parameter_constraints = {
        **InterestNoveltyKnowledgeBaseClassifier._parameter_constraints,
        "_decay_func_type": str,
        "_decay_func_factor": float
    }

    def __init__(self, *, learner_model: LearnerModel | None = None, threshold: float = 0.5, init_skill=0.,
                 def_var=0.5, beta: float = 0.5, positive_only=True, draw_proba_type: str = "dynamic",
                 draw_proba_static: float = 0.5, draw_proba_factor: float = 0.1,
                 decay_func_type: str = "short", decay_func_factor: float = 0.) -> None:
        super().__init__(learner_model=learner_model, threshold=threshold, init_skill=init_skill,
                         def_var=def_var, beta=beta, positive_only=positive_only, draw_proba_type=draw_proba_type,
                         draw_proba_static=draw_proba_static, draw_proba_factor=draw_proba_factor)

        if decay_func_type not in ("short", "long"):
            raise ValueError(
                f"The decay_func_type must be either short or long. Got {decay_func_type} instead."
            )
        self._decay_func_type = decay_func_type
        self._decay_func_factor = decay_func_factor

        self._validate_params()

    def __get_decay_func(self) -> Callable[[float], float]:
        """Get decay function based on decay_func_type.

        Returns
        -------
        Callable[float, float]
            The resulting decay_function.

        Notes
        -----
        Equations from: https://link.springer.com/article/10.1007/s11227-020-03266-2

        """
        if self._decay_func_type == "short":
            return lambda t_delta: min(2 / (1 + math.exp(self._decay_func_factor * t_delta)), 1.)

        return lambda t_delta: min(math.exp(-self._decay_func_factor * t_delta), 1.)

    # pylint: disable=too-many-locals
    def _update_knowledge_representation(self, x: EventModel, y: bool) -> None:
        if x.event_time is None:
            raise ValueError(
                "The event time should not be None when using InterestClassifier."
            )

        event_time_posix = dt.utcfromtimestamp(x.event_time)

        # make it a list because we need to use it more than one time later
        # select topic_kc_pairs with default event time = x.event_time
        learner_topic_kc_pairs = list(select_topic_kc_pairs(
            self._learner_model, x.knowledge, self._init_skill, self._def_var, x.event_time))
        learner_kcs = map(
            lambda topic_kc_pair: topic_kc_pair[1], learner_topic_kc_pairs)

        # apply interest decay
        decay_func = self.__get_decay_func()

        def __apply_interest_decay(kc: AbstractKnowledgeComponent) -> AbstractKnowledgeComponent:
            if kc.timestamp is None:
                raise ValueError(
                    "The timestamp field of knowledge component should not be None if using InterestClassifier."
                )
            t_delta = (event_time_posix -
                       dt.utcfromtimestamp(kc.timestamp)).days
            kc.update(mean=kc.mean * decay_func(float(t_delta)))
            return kc

        learner_kcs_decayed = map(__apply_interest_decay, learner_kcs)

        team_learner = self._gather_trueskill_team(learner_kcs_decayed)
        team_content = self._gather_trueskill_team(
            x.knowledge.knowledge_components())

        # learner always wins in interest
        updated_team_learner, _ = self._env.rate(
            [team_learner, team_content], ranks=[0, 1])

        for topic_kc_pair, rating in zip(learner_topic_kc_pairs, updated_team_learner):
            topic_id, kc = topic_kc_pair
            # need to update with timestamp=x.event_time as there are old kcs in the pairs
            kc.update(mean=rating.mean, variance=rating.sigma **
                      2, timestamp=x.event_time)
            self._learner_model.knowledge.update_kc(topic_id, kc)

    def predict_proba(self, x: EventModel) -> float:
        """Predict the probability of the learner's engagement in the given learning event.

        # TODO: describe the draw probability

        Parameters
        ----------
        x : EventModel
            A representation of a learning event.

        Returns
        -------
        float
            The probability that the learner engages in the given learning event.

        """
        learner_kcs = select_kcs(
            self._learner_model, x.knowledge, self._init_skill, self._def_var)
        content_kcs = x.knowledge.knowledge_components()
        return team_sum_quality(learner_kcs, content_kcs, self._beta)
