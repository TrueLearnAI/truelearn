from typing import Any, Optional, Dict
from typing_extensions import Final

from ._base import (
    InterestNoveltyKnowledgeBaseClassifier,
    team_sum_quality,
    select_kcs,
    select_topic_kc_pairs,
)
from truelearn.models import EventModel, LearnerModel


class KnowledgeClassifier(InterestNoveltyKnowledgeBaseClassifier):
    """A classifier that models the learner's knowledge and \
    makes prediction based on the knowledge.

    During the training process, the classifier uses the idea of game matching
    established in TrueSkill. It represents the learning process as a game of two teams.
    One team consists of all the knowledge components from the learnable unit and
    the other consist of all the corresponding knowledge components from the learner.
    Then, the classifier uses the given label to update the knowledge components
    of the learner.

    The update of knowledge components is based on the assumption that
    if the learner engages with the learnable unit, it means that
    the learner has a higher skill than the depth of the resource, which
    means that the learner wins the game.

    During the prediction process, the classifier uses cumulative density function of
    normal distribution to calculate the probability that the learner engage
    in the learning event. It calculates the probability of getting x in a
    Normal Distribution N(0, std) where x is the difference between
    the learner's skill (mean) and the learnable unit's skill (mean) and
    std is the standard deviation of the new normal distribution as a result of
    subtracting the two old normal distribution (learner and learnable unit).
    In TrueSkill's terminology, this calculates the win probability that
    the learner will win the content.
    """

    DRAW_PROBA_STATIC: Final[float] = 1e-9

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
        beta: float = 0.1,
        tau: float = 0.1,
        positive_only: bool = True,
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

        Returns:
            None
        """
        self._validate_params(
            learner_model=learner_model,
            threshold=threshold,
            init_skill=init_skill,
            def_var=def_var,
            tau=tau,
            beta=beta,
            positive_only=positive_only,
        )

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

    def _update_knowledge_representation(self, x: EventModel, y: bool) -> None:
        # make it a list because we need to use it more than one time later
        learner_topic_kc_pairs = list(
            select_topic_kc_pairs(
                self.learner_model,
                x.knowledge,
                self.init_skill,
                self.def_var,
            )
        )
        learner_kcs = map(
            lambda learner_topic_kc_pair: learner_topic_kc_pair[1],
            learner_topic_kc_pairs,
        )
        content_kcs = x.knowledge.knowledge_components()

        team_learner = self._gather_trueskill_team(learner_kcs)
        team_content = self._gather_trueskill_team(content_kcs)

        if y:
            # learner wins: lower rank == winning
            updated_team_learner, _ = self.__env.rate(
                [team_learner, team_content], ranks=[0, 1]
            )
        else:
            # content wins
            _, updated_team_learner = self.__env.rate(
                [team_content, team_learner], ranks=[0, 1]
            )

        for topic_kc_pair, rating in zip(learner_topic_kc_pairs, updated_team_learner):
            topic_id, kc = topic_kc_pair
            kc.update(mean=rating.mean, variance=rating.sigma**2)
            self.learner_model.knowledge.update_kc(topic_id, kc)

    def predict_proba(self, x: EventModel) -> float:
        learner_kcs = select_kcs(
            self.learner_model, x.knowledge, self.init_skill, self.def_var
        )
        content_kcs = x.knowledge.knowledge_components()
        return team_sum_quality(learner_kcs, content_kcs, self.beta)

    def get_learner_model(self) -> LearnerModel:
        """Get the learner model associated with this classifier.

        Returns:
            A learner model associated with this classifier.
        """
        return self.learner_model
