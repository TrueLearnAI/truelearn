from typing import Any, Optional, Dict

from truelearn.models import EventModel, LearnerModel
from ._base import (
    InterestNoveltyKnowledgeBaseClassifier,
    select_kcs,
    select_topic_kc_pairs,
)


class NoveltyClassifier(InterestNoveltyKnowledgeBaseClassifier):
    """A classifier that models the learner's knowledge and \
    makes prediction based on the knowledge.

    During the training process, the classifier uses the idea of game matching
    established in TrueSkill. It represents the learning process as a game of two teams.
    One team consists of all the knowledge components from the learnable unit and
    the other consist of all the corresponding knowledge components from the learner.
    Then, the classifier uses the given label to update the knowledge components of
    the learner.

    The update of knowledge components is based on the assumption that
    if the learner engages with the learnable unit, it means that
    the learner has skills similar to the depth of the resource, which
    means that the game is drawn.

    During the prediction process, the classifier uses the TrueSkill quality mechanism
    to evaluate the quality of the game. The idea is that if the quality of the game
    is high, it means that neither side can easily win the other.
    Thus, a high quality game means that learners are likely to engage with
    learnable units based on our assumption.
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
        beta: float = 0.1,
        tau: float = 0.1,
        positive_only: bool = True,
        draw_proba_type: str = "dynamic",
        draw_proba_static: float = 0.5,
        draw_proba_factor: float = 0.1,
    ) -> None:
        """Init NoveltyClassifier object.

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
            draw_proba_type:
                A str specifying the type of the draw probability.
                It could be either "static" or "dynamic". The "static"
                probability type requires an additional parameter
                draw_proba_static. The "dynamic" probability type calculates
                the draw probability based on the learner's previous engagement
                stats with educational resources.
            draw_proba_static:
                The global draw probability.
            draw_proba_factor:
                A factor that will be applied to both static and dynamic
                draw probability.

        Raises:
            ValueError: If draw_proba_type is neither "static" nor "dynamic".
        """
        self._validate_params(
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

    # pylint: disable=too-many-locals
    def _update_knowledge_representation(self, x: EventModel, y: bool) -> None:
        # make them list because we use them more than one time later
        learner_topic_kc_pairs = list(
            select_topic_kc_pairs(
                self.learner_model,
                x.knowledge,
                self.init_skill,
                self.def_var,
                x.event_time,
            )
        )
        learner_kcs = list(
            map(
                lambda learner_topic_kc_pair: learner_topic_kc_pair[1],
                learner_topic_kc_pairs,
            )
        )
        content_kcs = list(x.knowledge.knowledge_components())

        team_learner = self._gather_trueskill_team(learner_kcs)
        team_content = self._gather_trueskill_team(content_kcs)
        team_learner_mean = map(lambda learner_kc: learner_kc.mean, learner_kcs)
        team_content_mean = map(lambda content_kc: content_kc.mean, content_kcs)

        if y:
            # if learner wins
            # the game draws
            ranks = [0, 0]
        else:  # if the person is not engaged...
            difference = sum(team_learner_mean) - sum(team_content_mean)

            # check if the winner is learner or content,
            # uses the predicted skill representation
            if difference > 0.0:  # learner wins --> boring content
                ranks = [0, 1]
            elif difference < 0.0:  # learner loses --> intimidation
                ranks = [1, 0]
            else:
                ranks = None

        # update the rating based on the rank
        if ranks is not None:
            updated_team_learner, _ = self._env.rate(
                [team_learner, team_content], ranks=ranks
            )
        else:
            updated_team_learner = team_learner

        # update the learner's knowledge representation
        for topic_kc_pair, rating in zip(learner_topic_kc_pairs, updated_team_learner):
            topic_id, kc = topic_kc_pair
            kc.update(mean=rating.mu, variance=rating.sigma**2, timestamp=x.event_time,)
            self.learner_model.knowledge.update_kc(topic_id, kc)

    def predict_proba(self, x: EventModel) -> float:
        learner_kcs = select_kcs(
            self.learner_model, x.knowledge, self.init_skill, self.def_var, x.event_time
        )
        content_kcs = x.knowledge.knowledge_components()

        team_learner = self._gather_trueskill_team(learner_kcs)
        team_content = self._gather_trueskill_team(content_kcs)

        return self._env.quality([team_learner, team_content])

    def get_learner_model(self) -> LearnerModel:
        """Get the learner model associated with this classifier.

        Returns:
            A learner model associated with this classifier.
        """
        return self.learner_model
