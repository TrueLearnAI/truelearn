from __future__ import annotations

from ._base_classifier import InterestNoveltyKnowledgeBaseClassifier
from truelearn.models import EventModel


class NoveltyClassifier(InterestNoveltyKnowledgeBaseClassifier):
    """A Novelty Classifier.

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
        Train the model based on the given event and label.
    predict(x)
        Predict whether the learner will engage.
    predict_proba(x)
        Predict the probability of learner engagement.
    get_params()
        Get parameters associated with the model.

    """

    # pylint: disable=too-many-locals
    def _update_knowledge_representation(self, x: EventModel, y: bool) -> None:
        learner_topic_kc_pairs = list(self._select_topic_kc_pairs(x.knowledge))
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
        learner_kcs = map(
            lambda x: x[1], self._select_topic_kc_pairs(x.knowledge))
        content_kcs = x.knowledge.knowledge_components()

        team_learner = self._gather_trueskill_team(learner_kcs)
        team_content = self._gather_trueskill_team(content_kcs)

        return self._env.quality([team_learner, team_content])
