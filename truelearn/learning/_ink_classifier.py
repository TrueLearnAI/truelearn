from typing import Any, Union
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
        "_learner_model": MetaLearnerModel,
        "_threshold": float,
        "_k_init_skill": float,
        "_i_init_skill": float,
        "_k_def_var": float,
        "_i_def_var": float,
        "_k_beta": float,
        "_i_beta": float,
        "_tau": float,
        "_positive_only": bool,
        "_draw_proba_type": str,
        "_draw_proba_static": float,
        "_draw_proba_factor": float,
        "_decay_func_type": str,
        "_decay_func_factor": float,
        "_greedy": bool,
    }

    # pylint: disable=too-many-locals
    def __init__(
        self,
        *,
        learner_model: Union[MetaLearnerModel, None] = None,
        threshold: float = 0.5,
        k_init_skill: float = 0.0,
        i_init_skill: float = 0.0,
        k_def_var: float = 0.5,
        i_def_var: float = 0.5,
        k_beta: float = 0.1,
        i_beta: float = 0.1,
        tau: float = 0.5,
        positive_only: bool = True,
        draw_proba_type: str = "dynamic",
        draw_proba_static: float = 0.5,
        draw_proba_factor: float = 0.1,
        decay_func_type: str = "short",
        decay_func_factor: float = 0.0,
        greedy: bool = False,
    ) -> None:
        """Init INKClassifier object.

        Args:
            *:
                Use to reject positional arguments.
            learner_model:
                A representation of the learner.
            threshold:
                A float that determines the prediction threshold.
                When the predict is called, the classifier will return True iff
                the predicted probability is greater than the threshold.
            k_init_skill:
                The initial mean of the learner's knowledge/novelty.
            i_init_skill:
                The initial mean of the learner's interest.
            k_def_var:
                The initial variance of the learner's knowledge/novelty.
            i_def_var:
                The initial variance of the learner's interest.
            k_beta:
                The noise factor for NoveltyClassifier.
            i_beta:
                The noise factor for InterestClassifier.
            tau:
                The dynamic factor of learner's learning process.
            positive_only:
                A bool indicating whether the classifier only
                updates the learner's knowledge when encountering a positive label.
            draw_proba_type:
                A str specifying the type of the draw probability.
                It could be either "static" or "dynamic". The "static" probability type
                requires an additional parameter draw_proba_static.
                The "dynamic" probability
                type calculates the draw probability based on the learner's previous
                engagement stats with educational resources.
            draw_proba_static: The global draw probability.
            draw_proba_factor:
                A factor that will be applied to both
                static and dynamic draw probability.
            decay_func_type:
                A str specifying the type of the interest decay function.
                The allowed values are "short" and "long".
            decay_func_factor:
                A factor that will be used in both short and long
                interest decay function.
            greedy:
                A bool indicating whether the meta-learning should
                take the greedy approach. In the greedy approach,
                only incorrect predictions lead to the update of the weights.

        Raises:
            ValueError: If draw_proba_type is neither "static" nor "dynamic" or
                If decay_func_type is neither "short" nor "long".
        """
        if learner_model is None:
            learner_model = MetaLearnerModel()

        self._learner_model = learner_model
        self._threshold = threshold
        self._k_init_skill = k_init_skill
        self._i_init_skill = i_init_skill
        self._k_def_var = k_def_var
        self._i_def_var = i_def_var
        self._k_beta = k_beta
        self._i_beta = i_beta
        self._tau = tau
        self._positive_only = positive_only
        self._draw_proba_type = draw_proba_type
        self._draw_proba_static = draw_proba_static
        self._draw_proba_factor = draw_proba_factor
        self._decay_func_type = decay_func_type
        self._decay_func_factor = decay_func_factor
        self._greedy = greedy

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
            self._learner_model.novelty_weight["mean"],
            self._learner_model.novelty_weight["variance"],
        )
        mu_interest, var_interest = (
            self._learner_model.interest_weight["mean"],
            self._learner_model.interest_weight["variance"],
        )
        mu_bias, var_bias = (
            self._learner_model.bias_weight["mean"],
            self._learner_model.bias_weight["variance"],
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
            self._learner_model.novelty_weight["mean"],
            self._learner_model.novelty_weight["variance"],
        ) = (new_team_experts[0].mu, new_team_experts[0].sigma ** 2)
        (
            self._learner_model.interest_weight["mean"],
            self._learner_model.interest_weight["variance"],
        ) = (new_team_experts[1].mu, new_team_experts[1].sigma ** 2)
        (
            self._learner_model.bias_weight["mean"],
            self._learner_model.bias_weight["variance"],
        ) = (new_team_experts[2].mu, new_team_experts[2].sigma ** 2)

    def fit(self, x: EventModel, y: bool) -> Self:
        novelty_classifier = NoveltyClassifier(
            learner_model=self._learner_model.learner_novelty,
            threshold=self._threshold,
            init_skill=self._k_init_skill,
            def_var=self._k_def_var,
            tau=self._tau,
            beta=self._k_beta,
            positive_only=self._positive_only,
            draw_proba_type=self._draw_proba_type,
            draw_proba_static=self._draw_proba_static,
            draw_proba_factor=self._draw_proba_factor,
        )
        interest_classifier = InterestClassifier(
            learner_model=self._learner_model.learner_interest,
            threshold=self._threshold,
            init_skill=self._i_init_skill,
            def_var=self._i_def_var,
            tau=self._tau,
            beta=self._i_beta,
            positive_only=self._positive_only,
            draw_proba_type=self._draw_proba_type,
            draw_proba_static=self._draw_proba_static,
            draw_proba_factor=self._draw_proba_factor,
            decay_func_type=self._decay_func_type,
            decay_func_factor=self._decay_func_factor,
        )
        novelty_classifier.fit(x, y)
        interest_classifier.fit(x, y)

        pred_novelty = novelty_classifier.predict_proba(x)
        pred_interest = interest_classifier.predict_proba(x)
        self.__update_weights(pred_novelty, pred_interest, y)

        return self

    def predict(self, x: EventModel) -> bool:
        return self.predict_proba(x) > self._threshold

    def predict_proba(self, x: EventModel) -> float:
        novelty_classifier = NoveltyClassifier(
            learner_model=self._learner_model.learner_novelty,
            threshold=self._threshold,
            init_skill=self._k_init_skill,
            def_var=self._k_def_var,
            tau=self._tau,
            beta=self._k_beta,
            positive_only=self._positive_only,
            draw_proba_type=self._draw_proba_type,
            draw_proba_static=self._draw_proba_static,
            draw_proba_factor=self._draw_proba_factor,
        )
        interest_classifier = InterestClassifier(
            learner_model=self._learner_model.learner_interest,
            threshold=self._threshold,
            init_skill=self._i_init_skill,
            def_var=self._i_def_var,
            tau=self._tau,
            beta=self._i_beta,
            positive_only=self._positive_only,
            draw_proba_type=self._draw_proba_type,
            draw_proba_static=self._draw_proba_static,
            draw_proba_factor=self._draw_proba_factor,
            decay_func_type=self._decay_func_type,
            decay_func_factor=self._decay_func_factor,
        )

        mu_novelty, var_novelty = (
            self._learner_model.novelty_weight["mean"],
            self._learner_model.novelty_weight["variance"],
        )
        mu_interest, var_interest = (
            self._learner_model.interest_weight["mean"],
            self._learner_model.interest_weight["variance"],
        )
        mu_bias, var_bias = (
            self._learner_model.bias_weight["mean"],
            self._learner_model.bias_weight["variance"],
        )

        cur_pred = self.__calculate_sum_prediction(
            mu_novelty=mu_novelty,
            var_novelty=var_novelty,
            pred_novelty=novelty_classifier.predict_proba(x),
            mu_interest=mu_interest,
            var_interest=var_interest,
            pred_interest=interest_classifier.predict_proba(x),
            mu_bias=mu_bias,
            var_bias=var_bias,
            pred_bias=1.0,
        )

        return cur_pred
