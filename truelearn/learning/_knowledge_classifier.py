from typing_extensions import Final
from typing import Any

from ._base import (
    InterestNoveltyKnowledgeBaseClassifier,
    team_sum_quality,
    select_kcs,
    select_topic_kc_pairs,
)
from truelearn.models import EventModel, LearnerModel


class KnowledgeClassifier(InterestNoveltyKnowledgeBaseClassifier):
    """A Knowledge Classifier.

    # TODO: add description

    Parameters
    ----------
    learner_model: LearnerModel | None, optional
    threshold: float
        Threshold for judging learner engagement. If the probability of the learner engagement is greater
        than the threshold, the model will predict engagement.
    init_skill: float
        The initial skill (mean) of the learner given a new AbstractKnowledgeComponent.
    def_var: float
        The default variance of the new AbstractKnowledgeComponent.
    beta: float
        The noise factor, which is used in trueskill.
    positive_only: bool
        Whether the model updates itself only if encountering positive data.

    # TODO: this section should be moved to __init__ later

    Methods
    -------
    fit(x, y)
        Train the model based on the given event and label.
    predict(x)
        Predict whether the learner will engage.
    predict_proba(x)
        Predict the probability of learner engagement.
    get_params()
        Get the parameters associated with the model.
    set_params(**kargs)
        Set the parameters associated with the model.

    # TODO: remove method section after switching to google style

    """

    DRAW_PROBA_STATIC: Final[float] = 1e-9

    _parameter_constraints: dict[str, Any] = {
        **InterestNoveltyKnowledgeBaseClassifier._parameter_constraints,
    }

    def __init__(
        self,
        *,
        learner_model: LearnerModel | None = None,
        threshold: float = 0.5,
        init_skill: float = 0.0,
        def_var: float = 0.5,
        beta: float = 0.1,
        tau: float = 0.1,
        positive_only: bool = True,
    ) -> None:
        # the knowledge classifier doesn't rely on the draw probability
        # it utilizes different assumptions
        # so, we set draw probability to a very small value to avoid its impact
        super().__init__(
            learner_model=learner_model,
            threshold=threshold,
            init_skill=init_skill,
            def_var=def_var,
            tau=tau,
            beta=beta,
            positive_only=positive_only,
            draw_proba_type="static",
            draw_proba_static=KnowledgeClassifier.DRAW_PROBA_STATIC,
            draw_proba_factor=0.1,
        )

        self._validate_params()

    def _update_knowledge_representation(self, x: EventModel, y: bool) -> None:
        # make it a list because we need to use it more than one time later
        learner_topic_kc_pairs = list(
            select_topic_kc_pairs(
                self._learner_model,
                x.knowledge,
                self._init_skill,
                self._def_var,
            )
        )
        learner_kcs = map(
            lambda topic_kc_pair: topic_kc_pair[1], learner_topic_kc_pairs
        )
        content_kcs = x.knowledge.knowledge_components()

        team_learner = self._gather_trueskill_team(learner_kcs)
        team_content = self._gather_trueskill_team(content_kcs)

        if y:
            # learner wins: lower rank == winning
            updated_team_learner, _ = self._env.rate(
                [team_learner, team_content], ranks=[0, 1]
            )
        else:
            # content wins
            _, updated_team_learner = self._env.rate(
                [team_content, team_learner], ranks=[0, 1]
            )

        for topic_kc_pair, rating in zip(
            learner_topic_kc_pairs, updated_team_learner
        ):
            topic_id, kc = topic_kc_pair
            kc.update(mean=rating.mean, variance=rating.sigma**2)
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
        learner_kcs = select_kcs(
            self._learner_model, x.knowledge, self._init_skill, self._def_var
        )
        content_kcs = x.knowledge.knowledge_components()
        return team_sum_quality(learner_kcs, content_kcs, self._beta)
