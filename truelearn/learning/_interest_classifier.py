from typing import Callable, Any, Optional
from datetime import datetime as dt
import math

from ._base import (
    InterestNoveltyKnowledgeBaseClassifier,
    team_sum_quality,
    select_kcs,
    select_topic_kc_pairs,
)
from truelearn.models import (
    EventModel,
    LearnerModel,
    AbstractKnowledgeComponent,
)


class InterestClassifier(InterestNoveltyKnowledgeBaseClassifier):
    """A classifier that models the learner's interest and \
    makes prediction based on the interest.

    Note, the knowledge component in this context means
    the interest of the learner/learnable unit.

    During the training process, the classifier uses the idea of game matching
    established in TrueSkill. It represents the learning process as a game of two teams.
    One team consists of all the knowledge components from the learnable unit and
    the other consist of all the corresponding knowledge components from the learner.
    Then, the classifier uses the given label to update the knowledge components of
    the learner.

    The update of knowledge components is based on the assumption that
    if the learner engages with the learnable unit, it means that the learner
    has a higher skill than the depth of the resource, which means that
    the learner wins the game.

    During the prediction process, the classifier uses cumulative density function
    of normal distribution to calculate the probability that the learner engage in
    the learning event. It calculates the probability of getting x in a
    Normal Distribution N(0, std) where x is the difference between
    the learner's skill (mean) and the learnable unit's skill (mean) and
    std is the standard deviation of the new normal distribution as a result of
    subtracting the two old normal distribution (learner and learnable unit).
    In TrueSkill's terminology, this calculates the win probability that
    the learner will win the content.
    """

    _parameter_constraints: dict[str, Any] = {
        **InterestNoveltyKnowledgeBaseClassifier._parameter_constraints,
        "_decay_func_type": str,
        "_decay_func_factor": float,
    }

    def __init__(
        self,
        *,
        learner_model: Optional[LearnerModel] = None,
        threshold: float = 0.5,
        init_skill: float = 0.0,
        def_var: float = 0.5,
        beta: float = 0.1,
        tau: float = 0.1,
        positive_only: bool = True,
        draw_proba_type: str = "dynamic",
        draw_proba_static: float = 0.5,
        draw_proba_factor: float = 0.1,
        decay_func_type: str = "short",
        decay_func_factor: float = 0.0,
    ) -> None:
        """Init InterestClassifier object.

        Args:
            *:
                Use to reject positional arguments.
            learner_model:
                A representation of the learner.
            threshold:
                A float that determines the prediction threshold.
                When the predict is called, the classifier will return True iff
                the predicted probability is greater than the threshold.
            init_skill:
                The initial mean of the learner's knowledge component.
                It will be used when the learner interacts with knowledge components
                at its first time.
            def_var:
                The initial variance of the learner's knowledge component.
                It will be used when the learner interacts with knowledge components
                at its first time.
            beta:
                The noise factor.
            tau:
                The dynamic factor of learner's learning process.
                It's used to avoid the halting of the learning process.
            positive_only:
                A bool indicating whether the classifier only
                updates the learner's knowledge when encountering a positive label.
            decay_func_type:
                A str specifying the type of the interest decay function.
                The allowed values are "short" and "long".
            decay_func_factor:
                A factor that will be used in both short and long
                interest decay function.

        Raises:
            ValueError:
                If draw_proba_type is neither "static" nor "dynamic";
                If decay_func_type is neither "short" nor "long".
        """
        super().__init__(
            learner_model=learner_model,
            threshold=threshold,
            init_skill=init_skill,
            def_var=def_var,
            tau=tau,
            beta=beta,
            positive_only=positive_only,
            draw_proba_type=draw_proba_type,
            draw_proba_static=draw_proba_static,
            draw_proba_factor=draw_proba_factor,
        )

        if decay_func_type not in ("short", "long"):
            raise ValueError(
                f"The decay_func_type must be either short or long."
                f" Got {decay_func_type} instead."
            )
        self._decay_func_type = decay_func_type
        self._decay_func_factor = decay_func_factor

        self._validate_params()

    def __get_decay_func(self) -> Callable[[float], float]:
        """Get decay function based on decay_func_type.

        Returns:
            A decay function based on given type.

        Notes:
            Equations from: https://link.springer.com/article/10.1007/s11227-020-03266-2
        """
        if self._decay_func_type == "short":
            return lambda t_delta: min(
                2 / (1 + math.exp(self._decay_func_factor * t_delta)), 1.0
            )

        return lambda t_delta: min(math.exp(-self._decay_func_factor * t_delta), 1.0)

    # pylint: disable=too-many-locals
    def _update_knowledge_representation(self, x: EventModel, y: bool) -> None:
        if x.event_time is None:
            raise ValueError(
                "The event time should not be None when using InterestClassifier."
            )

        event_time_posix = dt.utcfromtimestamp(x.event_time)

        # make it a list because we need to use it more than one time later
        # select topic_kc_pairs with default event time = x.event_time
        learner_topic_kc_pairs = list(
            select_topic_kc_pairs(
                self._learner_model,
                x.knowledge,
                self._init_skill,
                self._def_var,
                x.event_time,
            )
        )
        learner_kcs = map(
            lambda topic_kc_pair: topic_kc_pair[1], learner_topic_kc_pairs
        )

        # apply interest decay
        decay_func = self.__get_decay_func()

        def __apply_interest_decay(
            kc: AbstractKnowledgeComponent,
        ) -> AbstractKnowledgeComponent:
            if kc.timestamp is None:
                raise ValueError(
                    "The timestamp field of knowledge component"
                    " should not be None if using InterestClassifier."
                )
            t_delta = (event_time_posix - dt.utcfromtimestamp(kc.timestamp)).days
            kc.update(mean=kc.mean * decay_func(float(t_delta)))
            return kc

        learner_kcs_decayed = map(__apply_interest_decay, learner_kcs)

        team_learner = self._gather_trueskill_team(learner_kcs_decayed)
        team_content = self._gather_trueskill_team(x.knowledge.knowledge_components())

        # learner always wins in interest
        updated_team_learner, _ = self._env.rate(
            [team_learner, team_content], ranks=[0, 1]
        )

        for topic_kc_pair, rating in zip(learner_topic_kc_pairs, updated_team_learner):
            topic_id, kc = topic_kc_pair
            # need to update with timestamp=x.event_time
            # as there are old kcs in the pairs
            kc.update(
                mean=rating.mean,
                variance=rating.sigma**2,
                timestamp=x.event_time,
            )
            self._learner_model.knowledge.update_kc(topic_id, kc)

    def predict_proba(self, x: EventModel) -> float:
        learner_kcs = select_kcs(
            self._learner_model, x.knowledge, self._init_skill, self._def_var
        )
        content_kcs = x.knowledge.knowledge_components()
        return team_sum_quality(learner_kcs, content_kcs, self._beta)
