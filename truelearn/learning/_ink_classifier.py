import math
from typing import Any, Optional, Dict, Tuple
from typing_extensions import Self, Final

import trueskill

from truelearn.models import EventModel, LearnerModel, LearnerMetaWeights
from truelearn.base import BaseClassifier
from ._base import team_sum_quality
from .._constraint import TypeConstraint
from ._novelty_classifier import NoveltyClassifier
from ._interest_classifier import InterestClassifier


class INKClassifier(BaseClassifier):
    """A meta-classifier that combines KnowledgeClassifier and InterestClassifier.

    During the training process, the meta-classifier individually trains
    the KnowledgeClassifier and the InterestClassifier. After that, the
    meta-classifier trains a set of weights by again using the ideas of team matching.
    One team consists of the weights of the knowledge, interest and bias, and
    the other team consists of the threshold. Then, the meta-classifier
    uses the given label to adjust the weights accordingly.

    During the prediction process, the meta-classifier individually uses the predict
    function of the KnowledgeClassifier and InterestClassifier.
    Then, it combines them by using the weights.

    Examples:
        >>> from truelearn.learning import INKClassifier
        >>> from truelearn.models import EventModel, Knowledge
        >>> from truelearn.models import KnowledgeComponent, LearnerMetaWeights
        >>> ink_classifier = INKClassifier()
        >>> ink_classifier
        INKClassifier()
        >>>
        >>> # use custom weights
        >>> weights = LearnerMetaWeights.Weights(mean=0.0, variance=0.5)
        >>> meta_weights = LearnerMetaWeights(novelty_weights=weights)
        >>> ink_classifier = INKClassifier(learner_meta_weights=meta_weights)
        >>>
        >>> # prepare an event model
        >>> knowledges = [
        ...     Knowledge({1: KnowledgeComponent(mean=0.15, variance=1e-9)}),
        ...     Knowledge({
        ...         2: KnowledgeComponent(mean=0.87, variance=1e-9),
        ...         3: KnowledgeComponent(mean=0.18, variance=1e-9),
        ...     }),
        ...     Knowledge({
        ...         1: KnowledgeComponent(mean=0.34, variance=1e-9),
        ...         3: KnowledgeComponent(mean=0.15, variance=1e-9),
        ...     }),
        ... ]
        >>> times = [0, 10514, 53621]
        >>> events = [
        ...     EventModel(knowledge, time)
        ...     for knowledge, time in zip(knowledges, times)
        ... ]
        >>> engage_stats = [True, False, True]
        >>> for event, engage_stats in zip(events, engage_stats):
        ...     ink_classifier = ink_classifier.fit(event, engage_stats)
        ...     print(
        ...         ink_classifier.predict(event),
        ...         ink_classifier.predict_proba(event)
        ...     )
        ...
        True 0.64105...
        False 0.44438...
        True 0.64909...
        >>> ink_classifier.get_params(deep=False)  # doctest:+ELLIPSIS
        {...'learner_meta_weights': LearnerMetaWeights(novelty_weights=Weights(\
mean=0.20461..., variance=0.45871...), interest_weights=Weights(\
mean=0.66315..., variance=0.42187...), bias_weights=Weights(\
mean=0.12698..., variance=0.39796...))...}

    """

    __DEFAULT_GLOBAL_SIGMA: Final[float] = 1e-9
    __DEFAULT_DRAW_PROBA: Final[float] = 1e-9

    _parameter_constraints: Dict[str, Any] = {
        **BaseClassifier._parameter_constraints,
        "learner_meta_weights": TypeConstraint(LearnerMetaWeights),
        "novelty_classifier": TypeConstraint(NoveltyClassifier),
        "interest_classifier": TypeConstraint(InterestClassifier),
        "threshold": TypeConstraint(float),
        "tau": TypeConstraint(float),
        "greedy": TypeConstraint(bool),
    }

    def __init__(
        self,
        *,
        learner_meta_weights: Optional[LearnerMetaWeights] = None,
        novelty_classifier: Optional[NoveltyClassifier] = None,
        interest_classifier: Optional[InterestClassifier] = None,
        threshold: float = 0.5,
        tau: float = 0.0,
        greedy: bool = False,
    ) -> None:
        """Init INKClassifier object.

        Args:
            *:
                Use to reject positional arguments.
            learner_meta_weights:
                The novelty/interest/bias weights.
            novelty_classifier:
                The NoveltyClassifier.
            interest_classifier:
                The InterestClassifier.
            threshold:
                A float that determines the classification threshold.
            tau:
                The dynamic factor of learner's learning process.
                It's used to avoid the halting of the learning process.
            greedy:
                A bool indicating whether meta-learning should
                take the greedy approach. In the greedy approach,
                only incorrect predictions lead to the update of the weights.

        Raises:
            TrueLearnTypeError:
                Types of parameters do not satisfy their constraints.
            TrueLearnValueError:
                Values of parameters do not satisfy their constraints.
        """
        self._novelty_classifier = novelty_classifier or NoveltyClassifier()
        self._interest_classifier = interest_classifier or InterestClassifier()
        self._learner_meta_weights = learner_meta_weights or LearnerMetaWeights()
        self._threshold = threshold
        self._tau = tau
        self._greedy = greedy

        self._validate_params()

    def __eval_matching_quality(
        self,
        *,
        novelty_weights: LearnerMetaWeights.Weights,
        pred_novelty: float,
        interest_weights: LearnerMetaWeights.Weights,
        pred_interest: float,
        bias_weights: LearnerMetaWeights.Weights,
        pred_bias: float,
    ) -> float:
        """Evaluate the matching quality of learner and content given the weights.

        Args:
            novelty_weights:
                A dict containing the mean and variance of the novelty_weights.
            pred_novelty:
                The predicted probability of the learner's engagement by using
                NoveltyClassifier.
            interest_weights:
                A dict containing the mean and variance of the interest_weights.
            pred_interest:
                The predicted probability of the learner's engagement by using
                InterestClassifier.
            bias_weights:
                A dict containing the mean and variance of the bias_weights.
            pred_bias:
                The predicted probability of the learner's engagement by using
                bias. This value is always 1.0.

        Returns:
            A float between [0, 1], indicating the matching quality
            of the learner and the content. The higher the value,
            the better the match.
        """
        team_learner_mean = [
            novelty_weights.mean * pred_novelty,
            interest_weights.mean * pred_interest,
            bias_weights.mean * pred_bias,
        ]
        team_learner_variance = [
            novelty_weights.variance * pred_novelty,
            interest_weights.variance * pred_interest,
            bias_weights.variance * pred_bias,
        ]
        team_content_mean = [self._threshold]
        team_content_variance = []

        return team_sum_quality(
            learner_mean=team_learner_mean,
            learner_variance=team_learner_variance,
            content_mean=team_content_mean,
            content_variance=team_content_variance,
            beta=0,
        )

    def __create_env(self):
        """Create the trueskill environment used in the training/prediction process."""
        return trueskill.TrueSkill(
            mu=0.0,
            sigma=INKClassifier.__DEFAULT_GLOBAL_SIGMA,
            beta=1,
            tau=self._tau,
            draw_probability=INKClassifier.__DEFAULT_DRAW_PROBA,
            backend="mpmath",
        )

    def __update_weights(
        self,
        x: EventModel,
        pred_novelty: float,
        pred_interest: float,
        pred_actual: float,
    ) -> None:
        """Update the weights of novelty, interest and bias.

        Args:
            x:
                A representation of the learning event.
            pred_novelty:
                The predicted probability of the learner's engagement by using
                NoveltyClassifier.
            pred_interest:
                The predicted probability of the learner's engagement by using
                InterestClassifier.
            pred_actual:
                Whether the learner actually engages in the given event. This value is
                either 0 or 1.
        """
        cur_pred = self.predict(x)

        # if the prediction is correct and greedy, don't train
        if self._greedy and cur_pred == pred_actual:
            return

        # train
        env = self.__create_env()
        team_experts = (
            env.create_rating(
                mu=self._learner_meta_weights.novelty_weights.mean,
                sigma=math.sqrt(self._learner_meta_weights.novelty_weights.variance),
            ),
            env.create_rating(
                mu=self._learner_meta_weights.interest_weights.mean,
                sigma=math.sqrt(self._learner_meta_weights.interest_weights.variance),
            ),
            env.create_rating(
                mu=self._learner_meta_weights.bias_weights.mean,
                sigma=math.sqrt(self._learner_meta_weights.bias_weights.variance),
            ),
        )

        team_threshold = (
            env.create_rating(
                mu=self._threshold, sigma=INKClassifier.__DEFAULT_GLOBAL_SIGMA
            ),
        )

        if pred_actual:  # weights need to be larger than a threshold
            new_team_experts, _ = env.rate(
                [team_experts, team_threshold],
                weights=[(pred_novelty, pred_interest, 1), (1,)],
                ranks=[0, 1],
            )
        else:
            new_team_experts, _ = env.rate(
                [team_experts, team_threshold],
                weights=[(pred_novelty, pred_interest, 1), (1,)],
                ranks=[1, 0],
            )

        # update skills
        self._learner_meta_weights.novelty_weights = LearnerMetaWeights.Weights(
            new_team_experts[0].mu, new_team_experts[0].sigma ** 2
        )
        self._learner_meta_weights.interest_weights = LearnerMetaWeights.Weights(
            new_team_experts[1].mu, new_team_experts[1].sigma ** 2
        )
        self._learner_meta_weights.bias_weights = LearnerMetaWeights.Weights(
            new_team_experts[2].mu, new_team_experts[2].sigma ** 2
        )

    def fit(self, x: EventModel, y: bool) -> Self:
        self._novelty_classifier.fit(x, y)
        self._interest_classifier.fit(x, y)

        pred_novelty = self._novelty_classifier.predict_proba(x)
        pred_interest = self._interest_classifier.predict_proba(x)

        self.__update_weights(x, pred_novelty, pred_interest, y)
        return self

    def predict(self, x: EventModel) -> bool:
        return self.predict_proba(x) > self._threshold

    def predict_proba(self, x: EventModel) -> float:
        return self.__eval_matching_quality(
            novelty_weights=self._learner_meta_weights.novelty_weights,
            pred_novelty=self._novelty_classifier.predict_proba(x),
            interest_weights=self._learner_meta_weights.interest_weights,
            pred_interest=self._interest_classifier.predict_proba(x),
            bias_weights=self._learner_meta_weights.bias_weights,
            pred_bias=1.0,
        )

    def get_learner_model(
        self,
    ) -> Tuple[LearnerModel, LearnerModel, LearnerMetaWeights]:
        """Get the learner model associated with this classifier.

        Returns:
            A tuple (novelty_learner, interest_learner, meta_weights) where
            novelty_learner is the learner model associated with the NoveltyClassifier,
            interest_learner is the learner model associated with the
            interestClassifier, meta_weights is the weights used in this classifier.
        """
        return (
            self._novelty_classifier.get_learner_model(),
            self._interest_classifier.get_learner_model(),
            self._learner_meta_weights,
        )
