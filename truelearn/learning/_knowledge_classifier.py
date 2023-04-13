from typing import Any, Optional, Dict, Iterable

import trueskill

from truelearn.models import LearnerModel, BaseKnowledgeComponent
from ._base import (
    InterestNoveltyKnowledgeBaseClassifier,
    gather_trueskill_team,
    team_sum_quality_from_kcs,
)


class KnowledgeClassifier(InterestNoveltyKnowledgeBaseClassifier):
    """A classifier that models the learner's knowledge and \
    makes prediction based on the knowledge.

    During the training process, the classifier uses the idea of game matching
    established in TrueSkill. It represents the learning process as a game of two teams.
    One team consists of all the knowledge components from the learnable unit, and
    the other consists of all the corresponding knowledge components from the learner.
    Then, the classifier uses the given label to update the knowledge components
    of the learner.

    The update of knowledge components is based on the assumption that
    if the learner engages with the learnable unit, it means that
    the learner has a higher skill than the depth of the resource, which
    means that the learner wins the game.

    During the prediction process, the classifier uses cumulative density function of
    normal distribution to calculate the probability that the learner engages
    in the learning event. It calculates the probability of getting x in a
    Normal Distribution N(0, std) where x is the difference between
    the learner's skill (mean) and the learnable unit's skill (mean) and
    std is the standard deviation of the new normal distribution as a result of
    subtracting the two old normal distribution (learner and learnable unit).
    In TrueSkill's terminology, this calculates the win probability that
    the learner will win the content.

    Examples:
        >>> from truelearn.learning import KnowledgeClassifier
        >>> from truelearn.models import EventModel, Knowledge, KnowledgeComponent
        >>> knowledge_classifier = KnowledgeClassifier()
        >>> knowledge_classifier
        KnowledgeClassifier()
        >>> # prepare an event model
        >>> knowledges = [
        ...     Knowledge({1: KnowledgeComponent(mean=0.57, variance=1e-9)}),
        ...     Knowledge({
        ...         2: KnowledgeComponent(mean=0.17, variance=1e-9),
        ...         3: KnowledgeComponent(mean=0.41, variance=1e-9),
        ...     }),
        ...     Knowledge({
        ...         1: KnowledgeComponent(mean=0.24, variance=1e-9),
        ...         3: KnowledgeComponent(mean=0.67, variance=1e-9),
        ...     }),
        ... ]
        >>> events = [EventModel(knowledge) for knowledge in knowledges]
        >>> engage_stats = [False, True, False]
        >>> for event, engage_stats in zip(events, engage_stats):
        ...     knowledge_classifier = knowledge_classifier.fit(event, engage_stats)
        ...     print(
        ...         knowledge_classifier.predict(event),
        ...         knowledge_classifier.predict_proba(event)
        ...     )
        ...
        False 0.21009...
        True 0.78306...
        False 0.36559...
        >>> knowledge_classifier.get_params()  # doctest:+ELLIPSIS
        {..., 'learner_model': LearnerModel(knowledge=Knowledge(knowledge={2: \
KnowledgeComponent(mean=0.60005..., variance=0.31394..., ...), ...}), ...}
    """

    _parameter_constraints: Dict[str, Any] = {
        **InterestNoveltyKnowledgeBaseClassifier._parameter_constraints,
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
        positive_only: bool = True,
        draw_proba_type: str = "dynamic",
        draw_proba_static: Optional[float] = None,
        draw_proba_factor: float = 0.1,
    ) -> None:
        """Init KnowledgeClassifier object.

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
            positive_only:
                A bool indicating whether the classifier only
                updates the learner's knowledge when encountering a positive label.
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

        Raises:
            TrueLearnTypeError:
                Types of parameters do not satisfy their constraints.
            TrueLearnValueError:
                Values of parameters do not satisfy their constraints.
        """
        # the knowledge classifier doesn't rely on the draw probability
        # it utilises different assumptions, so
        # we set draw probability to a very small value to avoid its impact
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

        self._validate_params()

    def _generate_ratings(  # pylint: disable=too-many-arguments
        self,
        env: trueskill.TrueSkill,
        learner_kcs: Iterable[BaseKnowledgeComponent],
        content_kcs: Iterable[BaseKnowledgeComponent],
        event_time: Optional[float],
        y: bool,
    ) -> Iterable[trueskill.Rating]:
        team_learner = gather_trueskill_team(env, learner_kcs)
        team_content = gather_trueskill_team(env, content_kcs)

        if y:
            # learner wins: lower rank == winning
            updated_team_learner, _ = env.rate(
                [team_learner, team_content], ranks=[0, 1]
            )
            return updated_team_learner

        # content wins
        _, updated_team_learner = env.rate([team_content, team_learner], ranks=[0, 1])
        return updated_team_learner

    def _eval_matching_quality(
        self,
        env: trueskill.TrueSkill,
        learner_kcs: Iterable[BaseKnowledgeComponent],
        content_kcs: Iterable[BaseKnowledgeComponent],
    ) -> float:
        return team_sum_quality_from_kcs(learner_kcs, content_kcs, self._beta)
