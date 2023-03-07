import math
from typing import Any, Optional, Dict
from typing_extensions import Self, Final

import trueskill
import mpmath

from truelearn.models import EventModel, LearnerMetaModel
from ._base import BaseClassifier
from ._novelty_classifier import NoveltyClassifier
from ._interest_classifier import InterestClassifier


class INKClassifier(BaseClassifier):
    """A meta-classifier that combines KnowledgeClassifier and InterestClassifier.

    During the training process, the meta-classifier individually trains
    the KnowledgeClassifier and the InterestClassifier. After that, the
    meta-classifier trains a set of weights by again using the ideas of team matching.
    One team consists of the weights of the knowledge, interest and bias and
    the other team consists of the threshold. Then, the meta-classifier
    uses the given label to adjust the weights accordingly.

    During the prediction process, the meta-classifier individually uses the predict
    function of the KnowledgeClassifier and InterestClassifier.
    Then, it combines them by using the weights.

    Examples:
        >>> from truelearn.learning import INKClassifier
        >>> from truelearn.models import EventModel, Knowledge, KnowledgeComponent
        >>> ink_classifier = INKClassifier()
        >>> ink_classifier
        INKClassifier()
        >>>
        >>> # use custom weights
        >>> weights = LearnerMetaModel.Weights(mean=0.0, variance=0.5)
        >>> ink_classifier = INKClassifier(novelty_weights=weights)
        >>>
        >>> # prepare event model
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
        True 0.5519515387629774
        False 0.3042337221090127
        True 0.6278686231266752
        >>> ink_classifier.get_params(deep=False)  # doctest:+ELLIPSIS
        {'bias_weights': Weights(mean=0.32119..., variance=0.88150...), ..., \
'interest_weights': Weights(mean=0.58194..., variance=1.07022...), ..., \
'novelty_weights': Weights(mean=0.39332..., variance=1.16897...), ...}
    """

    __DEFAULT_GLOBAL_SIGMA: Final[float] = 1e-9
    __DEFAULT_DRAW_PROBA: Final[float] = 1e-9

    _parameter_constraints: Dict[str, Any] = {
        **BaseClassifier._parameter_constraints,
        "novelty_classifier": NoveltyClassifier,
        "interest_classifier": InterestClassifier,
        "threshold": float,
        "tau": float,
        "greedy": bool,
        "novelty_weights": LearnerMetaModel.Weights,
        "interest_weights": LearnerMetaModel.Weights,
        "bias_weights": LearnerMetaModel.Weights,
    }

    def __init__(
        self,
        *,
        novelty_classifier: Optional[NoveltyClassifier] = None,
        interest_classifier: Optional[InterestClassifier] = None,
        threshold: float = 0.5,
        tau: float = 0.5,
        greedy: bool = False,
        novelty_weights: Optional[LearnerMetaModel.Weights] = None,
        interest_weights: Optional[LearnerMetaModel.Weights] = None,
        bias_weights: Optional[LearnerMetaModel.Weights] = None,
    ) -> None:
        """Init INKClassifier object.

        Args:
            *:
                Use to reject positional arguments.
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
                A bool indicating whether the meta-learning should
                take the greedy approach. In the greedy approach,
                only incorrect predictions lead to the update of the weights.
            novelty_weights:
                A Weights object containing the mean and variance.
            interest_weights:
                A Weights object containing the mean and variance.
            bias_weights:
                A Weights object containing the mean and variance.

        Raises:
            TypeError:
                Types of parameters mismatch their constraints.
            ValueError:
                If the parameter is not any of the valid values in the given tuple.
        """
        if novelty_classifier is None:
            novelty_classifier = NoveltyClassifier()
        if interest_classifier is None:
            interest_classifier = InterestClassifier()

        self.novelty_classifier = novelty_classifier
        self.interest_classifier = interest_classifier
        self.threshold = threshold
        self.tau = tau
        self.greedy = greedy

        if novelty_weights is None:
            novelty_weights = LearnerMetaModel.Weights()

        if interest_weights is None:
            interest_weights = LearnerMetaModel.Weights()

        if bias_weights is None:
            bias_weights = LearnerMetaModel.Weights()

        self.novelty_weights = novelty_weights
        self.interest_weights = interest_weights
        self.bias_weights = bias_weights

        self._validate_params()

    def __eval_matching_quality(
        self,
        *,
        novelty_weights: LearnerMetaModel.Weights,
        pred_novelty: float,
        interest_weights: LearnerMetaModel.Weights,
        pred_interest: float,
        bias_weights: LearnerMetaModel.Weights,
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
        difference = (
            (novelty_weights.mean * pred_novelty)
            + (interest_weights.mean * pred_interest)
            + (bias_weights.mean * pred_bias)
            - self.threshold
        )
        std = math.sqrt(
            novelty_weights.variance * pred_novelty
            + interest_weights.variance * pred_interest
            + bias_weights.variance * pred_bias
        )

        return float(mpmath.ncdf(difference, mu=0, sigma=std))

    def __create_env(self):
        """Create the trueskill environment used in the training/prediction process."""
        return trueskill.TrueSkill(
            mu=0.0,
            sigma=INKClassifier.__DEFAULT_GLOBAL_SIGMA,
            beta=1,
            tau=self.tau,
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
                Whether the learner actually engage in the given event. This value is
                either 0 or 1.
        """
        cur_pred = self.predict(x)

        # if prediction is correct and greedy, don't train
        if self.greedy and cur_pred == pred_actual:
            return

        # train
        env = self.__create_env()
        team_experts = (
            env.create_rating(
                mu=self.novelty_weights.mean,
                sigma=math.sqrt(self.novelty_weights.variance),
            ),
            env.create_rating(
                mu=self.interest_weights.mean,
                sigma=math.sqrt(self.interest_weights.variance),
            ),
            env.create_rating(
                mu=self.bias_weights.mean,
                sigma=math.sqrt(self.bias_weights.variance),
            ),
        )

        team_threshold = (
            env.create_rating(
                mu=self.threshold, sigma=INKClassifier.__DEFAULT_GLOBAL_SIGMA
            ),
        )

        if pred_actual:  # weights need to be larger than threshold
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
        self.novelty_weights = LearnerMetaModel.Weights(
            new_team_experts[0].mu, new_team_experts[0].sigma ** 2
        )
        self.interest_weights = LearnerMetaModel.Weights(
            new_team_experts[1].mu, new_team_experts[1].sigma ** 2
        )
        self.bias_weights = LearnerMetaModel.Weights(
            new_team_experts[2].mu, new_team_experts[2].sigma ** 2
        )

    def fit(self, x: EventModel, y: bool) -> Self:
        self.novelty_classifier.fit(x, y)
        self.interest_classifier.fit(x, y)

        pred_novelty = self.novelty_classifier.predict_proba(x)
        pred_interest = self.interest_classifier.predict_proba(x)

        self.__update_weights(x, pred_novelty, pred_interest, y)
        return self

    def predict(self, x: EventModel) -> bool:
        return self.predict_proba(x) > self.threshold

    def predict_proba(self, x: EventModel) -> float:
        return self.__eval_matching_quality(
            novelty_weights=self.novelty_weights,
            pred_novelty=self.novelty_classifier.predict_proba(x),
            interest_weights=self.interest_weights,
            pred_interest=self.interest_classifier.predict_proba(x),
            bias_weights=self.bias_weights,
            pred_bias=1.0,
        )

    def get_learner_model(self) -> LearnerMetaModel:
        """Get the learner model associated with this classifier.

        Returns:
            A learner model associated with this classifier.
        """
        return LearnerMetaModel(
            self.novelty_classifier.get_learner_model(),
            self.interest_classifier.get_learner_model(),
            self.novelty_weights,
            self.interest_weights,
            self.bias_weights,
        )
