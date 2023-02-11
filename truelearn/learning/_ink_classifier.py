from typing import Any
from typing_extensions import Self, Final
import math
import statistics

from ._base import BaseClassifier
from ._novelty_classifier import NoveltyClassifier
from ._interest_classifier import InterestClassifier
from truelearn.models import EventModel, MetaLearnerModel

import trueskill


class INKClassifier(BaseClassifier):
    """A INK (Interest, Novelty, Knowledge) Classifier.

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
        learner_model: MetaLearnerModel | None = None,
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
        mu_novelty,
        var_novelty,
        pred_novelty,
        mu_interest,
        var_interest,
        pred_interest,
        mu_bias,
        var_bias,
        pred_bias,
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
        self, pred_novelty: bool, pred_interest: bool, pred_actual: bool
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
            pred_bias=1,
        )

        # if prediction is correct and greedy, don't train
        if self._greedy and (cur_pred >= self._threshold) == pred_actual:
            return

        # train
        team_experts = (
            self._env.create_rating(
                mu=mu_novelty, sigma=math.sqrt(var_novelty)
            ),
            self._env.create_rating(
                mu=mu_interest, sigma=math.sqrt(var_interest)
            ),
            self._env.create_rating(mu=mu_bias, sigma=math.sqrt(var_bias)),
        )

        team_threshold = (
            self._env.create_rating(mu=0.5, sigma=INKClassifier.DEFAULT_SIGMA),
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
        """Train the model based on the given event and labels.

        Parameters
        ----------
        x : EventModel
            A representation of a learning event.
        y : bool
            A label that is either True or False.

        Returns
        -------
        Self
            The updated Classifier.

        """
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

        pred_novelty = novelty_classifier.predict(x)
        pred_interest = interest_classifier.predict(x)
        self.__update_weights(pred_novelty, pred_interest, y)

        return self

    def predict(self, x: EventModel) -> bool:
        return self.predict_proba(x) > self._threshold

    def predict_proba(self, x: EventModel) -> float:
        """Predict the probability of learner engagement in the given learning event.

        Parameters
        ----------
        x : EventModel
            A representation of a learning event.

        Returns
        -------
        float
            The probability that the learner will engage in the given learning event.

        """
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
            pred_novelty=novelty_classifier.predict(x),
            mu_interest=mu_interest,
            var_interest=var_interest,
            pred_interest=interest_classifier.predict(x),
            mu_bias=mu_bias,
            var_bias=var_bias,
            pred_bias=1,
        )

        return cur_pred
