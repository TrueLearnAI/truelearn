from typing import Any, Optional, Dict
from typing_extensions import Self, Final
import math
import statistics

from ._base import BaseClassifier
from ._novelty_classifier import NoveltyClassifier
from ._interest_classifier import InterestClassifier
from truelearn.models import EventModel, MetaLearnerModel

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

    _parameter_constraints: dict[str, Any] = {
        **BaseClassifier._parameter_constraints,
        "_threshold": float,
        "_tau": float,
        "_greedy": bool,
    }

    def __init__(
        self,
        *,
        novelty_classifier: Optional[NoveltyClassifier] = None,
        interest_classifier: Optional[InterestClassifier] = None,
        tau: float = 0.5,
        threshold: float = 0.5,
        greedy: bool = False,
        novelty_weight: Optional[Dict[str, float]] = None,
        interest_weight: Optional[Dict[str, float]] = None,
        bias_weight: Optional[Dict[str, float]] = None,
    ) -> None:
        """Init INKClassifier object.

        Args:
            *:
                Use to reject positional arguments.

            greedy:
                A bool indicating whether the meta-learning should
                take the greedy approach. In the greedy approach,
                only incorrect predictions lead to the update of the weights.

        Raises:
            ValueError:
                If draw_proba_type is neither "static" nor "dynamic" or
                If decay_func_type is neither "short" nor "long".
        """
        if novelty_classifier is None:
            novelty_classifier = NoveltyClassifier()
        if interest_classifier is None:
            interest_classifier = InterestClassifier()

        self._novelty_classifier = novelty_classifier
        self._interest_classifier = interest_classifier
        self._tau = tau
        self._threshold = threshold
        self._greedy = greedy

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

        self._novelty_weight: Dict[str, float] = novelty_weight
        self._interest_weight: Dict[str, float] = interest_weight
        self._bias_weight: Dict[str, float] = bias_weight

        self._env = trueskill.setup(
            mu=0.0,
            sigma=INKClassifier.DEFAULT_SIGMA,
            beta=1,
            tau=tau,
            draw_probability=INKClassifier.DEFAULT_DRAW_PROBA,
            backend="mpmath",
        )

        self._validate_params()

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
            - self._threshold
        )
        std = math.sqrt(
            ((var_novelty) * pred_novelty)
            + ((var_interest) * pred_interest)
            + ((var_bias) * pred_bias)
        )
        return statistics.NormalDist(mu=0, sigma=std).cdf(difference)

    def __update_weights(
        self, pred_novelty: float, pred_interest: float, pred_actual: float
    ) -> None:
        mu_novelty, var_novelty = (
            self._novelty_weight["mean"],
            self._novelty_weight["variance"],
        )
        mu_interest, var_interest = (
            self._interest_weight["mean"],
            self._interest_weight["variance"],
        )
        mu_bias, var_bias = (
            self._bias_weight["mean"],
            self._bias_weight["variance"],
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
        if self._greedy and (cur_pred >= self._threshold) == pred_actual:
            return

        # train
        team_experts = (
            self._env.create_rating(mu=mu_novelty, sigma=math.sqrt(var_novelty)),
            self._env.create_rating(mu=mu_interest, sigma=math.sqrt(var_interest)),
            self._env.create_rating(mu=mu_bias, sigma=math.sqrt(var_bias)),
        )

        team_threshold = (
            self._env.create_rating(
                mu=self._threshold, sigma=INKClassifier.DEFAULT_SIGMA
            ),
        )

        if pred_actual:  # weights need to be larger than threshold
            new_team_experts, _ = self._env.rate(
                [team_experts, team_threshold],
                weights=[(pred_novelty, pred_interest, 1), (1,)],
                ranks=[0, 1],
            )
        else:
            new_team_experts, _ = self._env.rate(
                [team_experts, team_threshold],
                weights=[(pred_novelty, pred_interest, 1), (1,)],
                ranks=[1, 0],
            )

        # update skills
        (
            self._novelty_weight["mean"],
            self._novelty_weight["variance"],
        ) = (new_team_experts[0].mu, new_team_experts[0].sigma ** 2)
        (
            self._interest_weight["mean"],
            self._interest_weight["variance"],
        ) = (new_team_experts[1].mu, new_team_experts[1].sigma ** 2)
        (
            self._bias_weight["mean"],
            self._bias_weight["variance"],
        ) = (new_team_experts[2].mu, new_team_experts[2].sigma ** 2)

    def fit(self, x: EventModel, y: bool) -> Self:
        self._novelty_classifier.fit(x, y)
        self._interest_classifier.fit(x, y)

        pred_novelty = self._novelty_classifier.predict_proba(x)
        pred_interest = self._interest_classifier.predict_proba(x)
        self.__update_weights(pred_novelty, pred_interest, y)

        return self

    def predict(self, x: EventModel) -> bool:
        return self.predict_proba(x) > self._threshold

    def predict_proba(self, x: EventModel) -> float:
        mu_novelty, var_novelty = (
            self._novelty_weight["mean"],
            self._novelty_weight["variance"],
        )
        mu_interest, var_interest = (
            self._interest_weight["mean"],
            self._interest_weight["variance"],
        )
        mu_bias, var_bias = (
            self._bias_weight["mean"],
            self._bias_weight["variance"],
        )

        cur_pred = self.__calculate_sum_prediction(
            mu_novelty=mu_novelty,
            var_novelty=var_novelty,
            pred_novelty=self._novelty_classifier.predict_proba(x),
            mu_interest=mu_interest,
            var_interest=var_interest,
            pred_interest=self._interest_classifier.predict_proba(x),
            mu_bias=mu_bias,
            var_bias=var_bias,
            pred_bias=1.0,
        )

        return cur_pred

    def get_learner_model(self) -> MetaLearnerModel:
        """Get the learner model associated with this classifier.

        Returns:
            A learner model associated with this classifier.
        """
        return MetaLearnerModel(
            self._novelty_classifier.get_learner_model(),
            self._interest_classifier.get_learner_model(),
            self._novelty_weight,
            self._interest_weight,
            self._bias_weight,
        )
