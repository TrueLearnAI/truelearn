from typing import Any, Optional, Dict
from typing_extensions import Self, Final
import math
import statistics

from ._base import BaseClassifier
from ._novelty_classifier import NoveltyClassifier
from ._interest_classifier import InterestClassifier
from truelearn.models import EventModel, LearnerMetaModel

import trueskill


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
    """

    DEFAULT_SIGMA: Final[float] = 1e-9
    DEFAULT_DRAW_PROBA: Final[float] = 1e-9

    _parameter_constraints: Dict[str, Any] = {
        **BaseClassifier._parameter_constraints,
        "novelty_classifier": [NoveltyClassifier, type(None)],
        "interest_classifier": [InterestClassifier, type(None)],
        "threshold": float,
        "tau": float,
        "greedy": bool,
        "novelty_weight": [dict, type(None)],
        "interest_weight": [dict, type(None)],
        "bias_weight": [dict, type(None)],
    }

    def __init__(
        self,
        *,
        novelty_classifier: Optional[NoveltyClassifier] = None,
        interest_classifier: Optional[InterestClassifier] = None,
        threshold: float = 0.5,
        tau: float = 0.5,
        greedy: bool = False,
        novelty_weight: Optional[Dict[str, float]] = None,
        interest_weight: Optional[Dict[str, float]] = None,
        bias_weight: Optional[Dict[str, float]] = None,
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
            novelty_weight:
                A dict containing the mean and variance of the novelty_weight.
            interest_weight:
                A dict containing the mean and variance of the interest_weight.
            bias_weight:
                A dict containing the mean and variance of the bias_weight.

        Raises:
            ValueError:
                If draw_proba_type is neither "static" nor "dynamic" or
                If decay_func_type is neither "short" nor "long".
        """
        self._validate_params(
            novelty_classifier=novelty_classifier,
            interest_classifier=interest_classifier,
            threshold=threshold,
            tau=tau,
            greedy=greedy,
            novelty_weight=novelty_weight,
            interest_weight=interest_weight,
            bias_weight=bias_weight,
        )

        if novelty_classifier is None:
            novelty_classifier = NoveltyClassifier()
        if interest_classifier is None:
            interest_classifier = InterestClassifier()

        self.novelty_classifier = novelty_classifier
        self.interest_classifier = interest_classifier
        self.threshold = threshold
        self.tau = tau
        self.greedy = greedy

        if novelty_weight is None:
            novelty_weight = {"mean": 0.0, "variance": 1.0}
        else:
            if "mean" not in novelty_weight or "variance" not in novelty_weight:
                raise ValueError(
                    f'The novelty_weight must contain "mean" and "variance".'
                    f" Got {novelty_weight} instead."
                )

        if interest_weight is None:
            interest_weight = {"mean": 0.0, "variance": 1.0}
        else:
            if "mean" not in interest_weight or "variance" not in interest_weight:
                raise ValueError(
                    f'The interest_weight must contain "mean" and "variance".'
                    f" Got {interest_weight} instead."
                )

        if bias_weight is None:
            bias_weight = {"mean": 0.0, "variance": 1.0}
        else:
            if "mean" not in bias_weight or "variance" not in bias_weight:
                raise ValueError(
                    f'The bias_weight must contain "mean" and "variance".'
                    f" Got {bias_weight} instead."
                )

        self.novelty_weight: Dict[str, float] = novelty_weight
        self.interest_weight: Dict[str, float] = interest_weight
        self.bias_weight: Dict[str, float] = bias_weight

        self.__env = trueskill.setup(
            mu=0.0,
            sigma=INKClassifier.DEFAULT_SIGMA,
            beta=1,
            tau=tau,
            draw_probability=INKClassifier.DEFAULT_DRAW_PROBA,
            backend="mpmath",
        )

    def __calculate_sum_prediction(
        self,
        *,
        mu_novelty: float,
        var_novelty: float,
        pred_novelty: float,
        mu_interest: float,
        var_interest: float,
        pred_interest: float,
        mu_bias: float,
        var_bias: float,
        pred_bias: float,
    ) -> float:
        difference = (
            (mu_novelty * pred_novelty)
            + (mu_interest * pred_interest)
            + (mu_bias * pred_bias)
            - self.threshold
        )
        std = math.sqrt(
            (var_novelty) * pred_novelty
            + (var_interest) * pred_interest
            + (var_bias) * pred_bias
        )
        return statistics.NormalDist(mu=0, sigma=std).cdf(difference)

    def __update_weights(
        self, pred_novelty: float, pred_interest: float, pred_actual: float
    ) -> None:
        mu_novelty, var_novelty = (
            self.novelty_weight["mean"],
            self.novelty_weight["variance"],
        )
        mu_interest, var_interest = (
            self.interest_weight["mean"],
            self.interest_weight["variance"],
        )
        mu_bias, var_bias = (
            self.bias_weight["mean"],
            self.bias_weight["variance"],
        )

        cur_pred = self.__calculate_sum_prediction(
            mu_novelty=mu_novelty,
            var_novelty=var_novelty,
            pred_novelty=pred_novelty,
            mu_interest=mu_interest,
            var_interest=var_interest,
            pred_interest=pred_interest,
            mu_bias=mu_bias,
            var_bias=var_bias,
            pred_bias=1.0,
        )

        # if prediction is correct and greedy, don't train
        if self.greedy and (cur_pred >= self.threshold) == pred_actual:
            return

        # train
        team_experts = (
            self.__env.create_rating(mu=mu_novelty, sigma=math.sqrt(var_novelty)),
            self.__env.create_rating(mu=mu_interest, sigma=math.sqrt(var_interest)),
            self.__env.create_rating(mu=mu_bias, sigma=math.sqrt(var_bias)),
        )

        team_threshold = (
            self.__env.create_rating(
                mu=self.threshold, sigma=INKClassifier.DEFAULT_SIGMA
            ),
        )

        if pred_actual:  # weights need to be larger than threshold
            new_team_experts, _ = self.__env.rate(
                [team_experts, team_threshold],
                weights=[(pred_novelty, pred_interest, 1), (1,)],
                ranks=[0, 1],
            )
        else:
            new_team_experts, _ = self.__env.rate(
                [team_experts, team_threshold],
                weights=[(pred_novelty, pred_interest, 1), (1,)],
                ranks=[1, 0],
            )

        # update skills
        (
            self.novelty_weight["mean"],
            self.novelty_weight["variance"],
        ) = (new_team_experts[0].mu, new_team_experts[0].sigma ** 2)
        (
            self.interest_weight["mean"],
            self.interest_weight["variance"],
        ) = (new_team_experts[1].mu, new_team_experts[1].sigma ** 2)
        (
            self.bias_weight["mean"],
            self.bias_weight["variance"],
        ) = (new_team_experts[2].mu, new_team_experts[2].sigma ** 2)

    def fit(self, x: EventModel, y: bool) -> Self:
        self.novelty_classifier.fit(x, y)
        self.interest_classifier.fit(x, y)

        pred_novelty = self.novelty_classifier.predict_proba(x)
        pred_interest = self.interest_classifier.predict_proba(x)
        self.__update_weights(pred_novelty, pred_interest, y)

        return self

    def predict(self, x: EventModel) -> bool:
        return self.predict_proba(x) > self.threshold

    def predict_proba(self, x: EventModel) -> float:
        mu_novelty, var_novelty = (
            self.novelty_weight["mean"],
            self.novelty_weight["variance"],
        )
        mu_interest, var_interest = (
            self.interest_weight["mean"],
            self.interest_weight["variance"],
        )
        mu_bias, var_bias = (
            self.bias_weight["mean"],
            self.bias_weight["variance"],
        )

        cur_pred = self.__calculate_sum_prediction(
            mu_novelty=mu_novelty,
            var_novelty=var_novelty,
            pred_novelty=self.novelty_classifier.predict_proba(x),
            mu_interest=mu_interest,
            var_interest=var_interest,
            pred_interest=self.interest_classifier.predict_proba(x),
            mu_bias=mu_bias,
            var_bias=var_bias,
            pred_bias=1.0,
        )

        return cur_pred

    def get_learner_model(self) -> LearnerMetaModel:
        """Get the learner model associated with this classifier.

        Returns:
            A learner model associated with this classifier.
        """
        return LearnerMetaModel(
            self.novelty_classifier.get_learner_model(),
            self.interest_classifier.get_learner_model(),
            self.novelty_weight,
            self.interest_weight,
            self.bias_weight,
        )
