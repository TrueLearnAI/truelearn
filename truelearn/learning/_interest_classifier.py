import math
from typing import Callable, Any, Optional, Dict, Iterable
from datetime import datetime as dt

import trueskill

from truelearn.errors import TrueLearnValueError
from truelearn.models import (
    LearnerModel,
    BaseKnowledgeComponent,
)
from ._base import (
    InterestNoveltyKnowledgeBaseClassifier,
    gather_trueskill_team,
    team_sum_quality_from_kcs,
)
from .._constraint import TypeConstraint, ValueConstraint, Range


class InterestClassifier(InterestNoveltyKnowledgeBaseClassifier):
    """A classifier that models the learner's interest and \
    makes prediction based on the interest.

    Note, the knowledge component in this context means
    the interest of the learner/learnable unit.

    During the training process, the classifier uses the idea of game matching
    established in TrueSkill. It represents the learning process as a game of two teams.
    One team consists of all the knowledge components from the learnable unit, and
    the other consists of all the corresponding knowledge components from the learner.
    Then, the classifier uses the given label to update the knowledge components of
    the learner.

    The update of knowledge components is based on the assumption that
    if the learner engages with the learnable unit, it means that the learner
    has a higher skill than the depth of the resource, which means that
    the learner wins the game.

    During the prediction process, the classifier uses cumulative density function
    of normal distribution to calculate the probability that the learner engages in
    the learning event. It calculates the probability of getting x in a
    Normal Distribution N(0, std) where x is the difference between
    the learner's skill (mean) and the learnable unit's skill (mean) and
    std is the standard deviation of the new normal distribution as a result of
    subtracting the two old normal distribution (learner and learnable unit).
    In TrueSkill's terminology, this calculates the win probability that
    the learner will win the content.

    Examples:
        >>> from truelearn.learning import InterestClassifier
        >>> from truelearn.models import EventModel, Knowledge, KnowledgeComponent
        >>> interest_classifier = InterestClassifier()
        >>> interest_classifier
        InterestClassifier()
        >>> # prepare an event model
        >>> knowledges = [
        ...     Knowledge({1: KnowledgeComponent(mean=0.57, variance=1e-9)}),
        ...     Knowledge({
        ...         2: KnowledgeComponent(mean=0.07, variance=1e-9),
        ...         3: KnowledgeComponent(mean=0.18, variance=1e-9),
        ...     }),
        ...     Knowledge({
        ...         1: KnowledgeComponent(mean=0.34, variance=1e-9),
        ...         3: KnowledgeComponent(mean=0.15, variance=1e-9),
        ...     }),
        ... ]
        >>> times = [0, 1024, 5381]
        >>> events = [
        ...     EventModel(knowledge, time)
        ...     for knowledge, time in zip(knowledges, times)
        ... ]
        >>> engage_stats = [False, True, False]
        >>> for event, engage_stats in zip(events, engage_stats):
        ...     interest_classifier = interest_classifier.fit(event, engage_stats)
        ...     print(
        ...         interest_classifier.predict(event),
        ...         interest_classifier.predict_proba(event)
        ...     )
        ...
        True 0.88450...
        True 0.81079...
        True 0.95872...
        >>> interest_classifier.get_params()  # doctest:+ELLIPSIS
        {..., 'learner_model': LearnerModel(knowledge=Knowledge(knowledge=\
{1: KnowledgeComponent(mean=0.99556..., variance=0.10483..., ...), ...}), ...}
    """

    _parameter_constraints: Dict[str, Any] = {
        **InterestNoveltyKnowledgeBaseClassifier._parameter_constraints,
        "decay_func_type": ValueConstraint("short", "long"),
        "decay_func_factor": [
            TypeConstraint(float),
            ValueConstraint(Range(ge=0)),
        ],
    }

    def __init__(
        self,
        *,
        learner_model: Optional[LearnerModel] = None,
        threshold: float = 0.5,
        init_skill: float = 0.0,
        def_var: float = 0.5,
        beta: float = 0.0,
        tau: float = 0.0,
        draw_proba_type: str = "dynamic",
        draw_proba_static: Optional[float] = None,
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
                A float that determines the classification threshold.
            init_skill:
                The initial mean of the learner's knowledge component.
                It will be used when the learner interacts with knowledge components
                for the first time.
            def_var:
                The initial variance (>0) of the learner's knowledge component.
                It will be used when the learner interacts with knowledge components
                for the first time.
            beta:
                The distance which guarantees about 76% chance of winning.
                The recommended value is sqrt(def_var) / 2.
            tau:
                The dynamic factor of learner's learning process.
                It's used to avoid the halting of the learning process.
            draw_proba_type:
                A str specifying the type of the draw probability.
                It could be either "static" or "dynamic". The "static"
                probability type requires an additional parameter
                draw_proba_static. The "dynamic" probability type calculates
                the draw probability based on the learner's previous engagement
                stats with educational resources.
            draw_proba_static:
                The global draw probability (>=0).
            draw_proba_factor:
                A factor (>=0) that will be applied to both static and dynamic
                draw probability.
            decay_func_type:
                A str specifying the type of the interest decay function.
                The allowed values are "short" and "long".
            decay_func_factor:
                A factor (>=0) that will be used in both short and long
                interest decay function. Defaults to 0, which disables
                the interest decay function.

        Raises:
            TrueLearnTypeError:
                Types of parameters do not satisfy their constraints.
            TrueLearnValueError:
                Values of parameters do not satisfy their constraints.
        """
        super().__init__(
            learner_model=learner_model,
            threshold=threshold,
            init_skill=init_skill,
            def_var=def_var,
            tau=tau,
            beta=beta,
            # learner always wins in interest classifier,
            # hence we should always update regardless of the actual label
            # positive_only should be disabled to ensure the update method
            # is always called
            positive_only=False,
            draw_proba_type=draw_proba_type,
            draw_proba_static=draw_proba_static,
            draw_proba_factor=draw_proba_factor,
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

    def _generate_ratings(
        self,
        env: trueskill.TrueSkill,
        learner_kcs: Iterable[BaseKnowledgeComponent],
        content_kcs: Iterable[BaseKnowledgeComponent],
        event_time: Optional[float],
        _y: bool,
    ) -> Iterable[trueskill.Rating]:
        """Generate an iterable of the updated Rating for the learner.

        The Rating is generated based on the label and optionally
        event_time (for InterestClassifier).

        Args:
            env:
                The trueskill environment where the training/prediction happens.
            learner_kcs:
                An iterable of learner's knowledge components.
            content_kcs:
                An iterable of content's knowledge components.
            event_time:
                An optional float representing the event time.
            _y:
                A bool indicating whether the learner engages in
                the learning event.

        Returns:
            An iterable of trueskill.Rating.

        Raises:
            TrueLearnValueError:
                If the event timestamp is None or any timestamp of
                the learner's knowledge components is None.
        """
        if event_time is None:
            raise TrueLearnValueError(
                "The event time should not be None when using InterestClassifier."
            )

        event_time_posix = dt.utcfromtimestamp(event_time)

        # apply interest decay
        decay_func = self.__get_decay_func()

        def __apply_interest_decay(
            learner_kc: BaseKnowledgeComponent,
        ) -> BaseKnowledgeComponent:
            if learner_kc.timestamp is None:
                raise TrueLearnValueError(
                    "The timestamp field of knowledge component"
                    " should not be None if using InterestClassifier."
                )
            t_delta = (
                event_time_posix - dt.utcfromtimestamp(learner_kc.timestamp)
            ).days
            learner_kc.update(mean=learner_kc.mean * decay_func(float(t_delta)))
            return learner_kc

        learner_kcs_decayed = map(__apply_interest_decay, learner_kcs)

        team_learner = gather_trueskill_team(env, learner_kcs_decayed)
        team_content = gather_trueskill_team(env, content_kcs)

        # learner always wins in interest
        updated_team_learner, _ = env.rate([team_learner, team_content], ranks=[0, 1])
        return updated_team_learner

    def _eval_matching_quality(
        self,
        env: trueskill.TrueSkill,
        learner_kcs: Iterable[BaseKnowledgeComponent],
        content_kcs: Iterable[BaseKnowledgeComponent],
    ) -> float:
        return team_sum_quality_from_kcs(learner_kcs, content_kcs, self._beta)
