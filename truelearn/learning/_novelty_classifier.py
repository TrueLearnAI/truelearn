from __future__ import annotations

from ._base import InterestNoveltyKnowledgeBaseClassifier, select_kcs, select_topic_kc_pairs
from truelearn.models import EventModel, LearnerModel


class NoveltyClassifier(InterestNoveltyKnowledgeBaseClassifier):
    """A Novelty Classifier.

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

    def __init__(self, *, learner_model: LearnerModel | None = None, threshold: float = 0.5, init_skill=0.,
                 def_var=0.5, beta: float = 0.5, positive_only=True, draw_proba_type: str = "dynamic",
                 draw_proba_static: float = 0.5, draw_proba_factor: float = 0.1) -> None:
        super().__init__(learner_model=learner_model, threshold=threshold, init_skill=init_skill,
                         def_var=def_var, beta=beta, positive_only=positive_only, draw_proba_type=draw_proba_type,
                         draw_proba_static=draw_proba_static, draw_proba_factor=draw_proba_factor)

    # pylint: disable=too-many-locals
    def _update_knowledge_representation(self, x: EventModel, y: bool) -> None:
        # make them list because we use them more than one time later
        learner_topic_kc_pairs = list(select_topic_kc_pairs(
            self._learner_model, x.knowledge, self._init_skill, self._def_var))
        learner_kcs = list(
            map(
                lambda topic_kc_pair: topic_kc_pair[1],
                learner_topic_kc_pairs
            )
        )
        content_kcs = list(x.knowledge.knowledge_components())

        team_learner = self._gather_trueskill_team(learner_kcs)
        team_content = self._gather_trueskill_team(content_kcs)
        team_learner_mean = map(lambda kc: kc.mean, learner_kcs)
        team_content_mean = map(lambda kc: kc.mean, content_kcs)

        if y:
            # if learner wins, use pos_learner skill which is updated with them topics ;)
            ranks = [0, 0]
        else:  # if the person is not engaged...
            difference = sum(team_learner_mean) - sum(team_content_mean)

            # check if the winner is learner or content, uses the predicted skill representation
            if difference > 0.:  # learner wins --> boring content
                ranks = [0, 1]
            elif difference < 0.:  # learner loses --> intimidation
                ranks = [1, 0]
            else:
                ranks = None

        # update the rating based on the rank
        if ranks is not None:
            updated_team_learner, _ = self._env.rate(
                [team_learner, team_content], ranks=ranks)
        else:
            updated_team_learner = team_learner

        # update the learner's knowledge representation
        for topic_kc_pair, rating in zip(learner_topic_kc_pairs, updated_team_learner):
            topic_id, kc = topic_kc_pair
            kc.update(rating.mean, rating.sigma ** 2)
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

        team_learner = self._gather_trueskill_team(learner_kcs)
        team_content = self._gather_trueskill_team(content_kcs)

        return self._env.quality([team_learner, team_content])
