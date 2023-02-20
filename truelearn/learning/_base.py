import math
import collections
from abc import ABC, abstractmethod
from typing import Iterable, Hashable, Any, Optional, Tuple, Dict
from typing_extensions import Self, Final, final

import trueskill
import mpmath

from truelearn.models import (
    EventModel,
    Knowledge,
    AbstractKnowledgeComponent,
    LearnerModel,
)


# pylint: disable=pointless-string-statement
"""
Copyright of `get_params`, `set_params` are held by
[BSD 3-Clause License, scikit-learn developers, 2007-2022].
"""


class BaseClassifier(ABC):
    """The base class of all the classifiers in TrueLearn.

    Every subclass of the BaseClassifier should define their
    `_parameter_constraints`. This specified the parameters exposed
    via the `get_params` method and the constraints on the type of
    the parameters.

    The `_parameter_constraints` is a dictionary that maps parameter
    names to its expected type. The expected type can be a list or a single type
    as it's possible for a type to accept more than one type.
    To do the constraint check based on this, simply call `self._validate_params`
    in your classifier.
    """

    __DEEP_PARAM_DELIMITER: Final[str] = "__"
    _parameter_constraints: Dict[str, Any] = {}

    @abstractmethod
    def fit(self, x: EventModel, y: bool) -> Self:
        """Train the model.

        Args:
            x: A representation of a learning event.
            y: A bool indicating whether the learner engages in the learning event.

        Returns:
            The updated classifier object.
        """

    @abstractmethod
    def predict(self, x: EventModel) -> bool:
        """Predict whether the learner will engage in the learning event.

        Args:
            x: A representation of a learning event.

        Returns:
            A bool indicating whether the learner will engage in the learning event.
        """

    @abstractmethod
    def predict_proba(self, x: EventModel) -> float:
        """Predict the probability that the learner will engage in the learning event.

        Args:
            x: A representation of a learning event.

        Returns:
            A float indicating the probability that the learner will engage
            in the learning event.
        """

    @final
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this Classifier.

        Args:
            deep:
                If True, will return the parameters for this Classifier and
                contained sub-objects that inherits BaseClassifier class.

        Returns:
            A dict mapping variable names to the corresponding objects.
        """
        param_names = list(self._parameter_constraints.keys())
        param_names.sort()

        out = {}
        for key in param_names:
            if not hasattr(self, key):
                raise ValueError(
                    f"The specified parameter name {key}"
                    f" is not in the {self.__class__.__name__}."
                )

            value = getattr(self, key)
            if deep and isinstance(value, BaseClassifier):
                deep_items = value.get_params().items()
                out.update(
                    (key + BaseClassifier.__DEEP_PARAM_DELIMITER + k, val)
                    for k, val in deep_items
                )
            out[key] = value

        return out

    @final
    def set_params(self, **args) -> Self:
        """Set the parameters of this Classifier.

        A value can be reset only if the given parameter
        has the same type as the original value.

        Args:
          **args:
            Keyword arguments.
            The key should match the parameter names of the classifier.
            The arguments should have the correct type.

        Returns:
            The updated classifier.

        Raises:
            TypeError:
                If the given value doesn't have the same type
                as the original value.
            ValueError:
                If the given argument name is not in the class.
        """
        # avoid running `self.get_params` if there is no given params
        if not args:
            return self

        valid_params = self.get_params(deep=True)

        # a dictionary that stores params for nested classifiers
        # it stores a map from nested_classifier_name to its parameters (a dict)
        # { nested_classifier_name => {key => value} }
        nested_params = collections.defaultdict(dict)

        for key, value in args.items():
            key, delim, sub_key = key.partition(BaseClassifier.__DEEP_PARAM_DELIMITER)
            if key not in valid_params:
                raise ValueError(
                    f"The given parameter {key}"
                    f" is not in the class {self.__class__.__name__}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                # ensure that they have the same type
                if not isinstance(value, type(valid_params[key])):
                    raise TypeError(
                        f"The given parameter {key}"
                        " doesn't have the same type"
                        " as the original value."
                        f" Expected {type(valid_params[key])}."
                        f" Got {type(value)}."
                    )
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    @final
    def _validate_params(self, **kargs) -> None:
        """Validate types of given arguments in __init__.

        Args:
            **kargs: A dict of (param_name, param_value) pair.

        Raises:
            TypeError:
                param_value or value in _parameter_constraints are
                not valid types.
        """
        for (
            param_name,
            param_value,
        ) in kargs.items():
            # ensure param_name is in the valid params dictionary
            if param_name not in self._parameter_constraints:
                continue

            expected_param_type = self._parameter_constraints.get(param_name)

            # ensure expected_param_type is properly set
            # `isinstance(expected_param_type, type)` works for python 3
            if isinstance(expected_param_type, list):
                # check if all the element inside the list are classes
                if not all(
                    isinstance(param_type_unpacked, type)
                    for param_type_unpacked in expected_param_type
                ):
                    raise TypeError(
                        "The given constraint list contains non-class element."
                    )
            else:
                # check if expected_param_type is a class
                if not isinstance(expected_param_type, type):
                    raise TypeError(
                        f"The given constraint {expected_param_type} is not a class."
                    )

            if isinstance(expected_param_type, list):
                # if it matches none of the types in the constraints
                if not any(
                    isinstance(param_value, param_type_unpacked)
                    for param_type_unpacked in list(expected_param_type)
                ):
                    param_classname_expected = list(
                        map(lambda cls: cls.__name__, expected_param_type)
                    )
                    raise TypeError(
                        f"The {param_name} parameter of {self.__class__.__name__}"
                        f" __init__ function must be one of the classes"
                        f" in {param_classname_expected}."
                        f" Got {param_value.__class__.__name__} instead."
                    )
            else:
                if not isinstance(param_value, expected_param_type):
                    raise TypeError(
                        f"The {param_name} parameter of {self.__class__.__name__}"
                        f" must be {expected_param_type.__name__}."
                        f" Got {param_value.__class__.__name__} instead."
                    )


class InterestNoveltyKnowledgeBaseClassifier(BaseClassifier):
    """A Base Classifier for KnowledgeClassifier, NoveltyClassifier \
    and InterestClassifier.

    It defines the necessary instance variables and
    common methods to interact with the LearnerModel.
    """

    DEFAULT_CONTENT_SIGMA: Final[float] = 1e-9
    DEFAULT_DRAW_PROBA_LOW: Final[float] = 1e-9
    DEFAULT_DRAW_PROBA_HIGH: Final[float] = 0.999999999

    _parameter_constraints: Dict[str, Any] = {
        **BaseClassifier._parameter_constraints,
        "learner_model": [LearnerModel, type(None)],
        "threshold": float,
        "init_skill": float,
        "def_var": float,
        "tau": float,
        "beta": float,
        "positive_only": bool,
        "draw_proba_type": str,
        "draw_proba_static": [float, type(None)],
        "draw_proba_factor": float,
    }

    def __init__(
        self,
        *,
        learner_model: Optional[LearnerModel],
        threshold: float,
        init_skill: float,
        def_var: float,
        beta: float,
        tau: float,
        positive_only: bool,
        draw_proba_type: str,
        draw_proba_static: Optional[float],
        draw_proba_factor: float,
    ) -> None:
        """Init InterestNoveltyKnowledgeBaseClassifier object.

        Args:
            *:
                Use to reject positional arguments.
            learner_model:
                A representation of the learner.
            threshold:
                A float that determines the classification threshold.
            init_skill:
                The initial mean of the learner's knowledge component.
                It will be used when the learner interacts with some
                knowledge components at its first time.
            def_var:
                The initial variance of the learner's knowledge component.
                It will be used when the learner interacts with some
                knowledge components at its first time.
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
                It could be either "static" or "dynamic". The "static" probability type
                requires an additional parameter draw_proba_static.
                The "dynamic" probability type calculates the draw probability
                based on the learner's previous engagement stats
                with educational resources.
            draw_proba_static:
                The global draw probability.
            draw_proba_factor:
                A factor that will be applied to both
                static and dynamic draw probability.

        Raises:
            ValueError: If draw_proba_type is neither "static" nor "dynamic".
        """
        super().__init__()

        if learner_model is None:
            self.learner_model = LearnerModel()
        else:
            self.learner_model = learner_model
        self.threshold = threshold
        self.init_skill = init_skill
        self.def_var = def_var
        self.tau = tau
        self.beta = beta
        self.positive_only = positive_only

        if draw_proba_type not in ("static", "dynamic"):
            raise ValueError(
                f"The draw_proba_type should be either static or dynamic."
                f" Got {draw_proba_type} instead."
            )

        self.draw_proba_type = draw_proba_type
        self.draw_proba_factor = draw_proba_factor
        self.draw_proba_static = draw_proba_static

        self.__setup_env()

    @final
    def __calculate_draw_proba(self) -> float:
        if self.draw_proba_type == "static":
            # delayed check as this can be potentially replaced by set_params
            if self.draw_proba_static is None:
                raise ValueError(
                    "When draw_proba_type is set to static,"
                    " the draw_proba_static should not be None."
                )
            return self.draw_proba_static * self.draw_proba_factor

        total_engagement_stats = max(
            1,
            self.learner_model.number_of_engagements
            + self.learner_model.number_of_non_engagements,
        )
        draw_probability = float(
            self.learner_model.number_of_engagements / total_engagement_stats
        )

        # clamp the value between [DEFAULT_DRAW_PROBA_LOW, DEFAULT_DRAW_PROBA_HIGH]
        draw_probability = max(
            min(
                InterestNoveltyKnowledgeBaseClassifier.DEFAULT_DRAW_PROBA_HIGH,
                draw_probability,
            ),
            InterestNoveltyKnowledgeBaseClassifier.DEFAULT_DRAW_PROBA_LOW,
        )

        # draw_proba_param is a factor if the type is dynamic
        return draw_probability * self.draw_proba_factor

    @final
    def __setup_env(self) -> None:
        """Setup the trueskill environment used in the training process."""
        self.draw_probability = self.__calculate_draw_proba()
        self._env = trueskill.TrueSkill(
            mu=0.0,
            sigma=InterestNoveltyKnowledgeBaseClassifier.DEFAULT_CONTENT_SIGMA,
            beta=self.beta,
            tau=self.tau,
            draw_probability=self.draw_probability,
            backend="mpmath"
        )

    @final
    def __update_engagement_stats(self, y: bool) -> None:
        """Update the learner's engagement stats based on the given label.

        Args:
            y: A bool indicating whether the learner engage in the learning event.
        """
        if y:
            self.learner_model.number_of_engagements += 1
        else:
            self.learner_model.number_of_non_engagements += 1

    @final
    def _gather_trueskill_team(
        self, kcs: Iterable[AbstractKnowledgeComponent]
    ) -> Tuple[trueskill.Rating, ...]:
        """Return a tuple of trueskill Rating \
        created from the given iterable of knowledge components.

        Args:
            kcs: An iterable of knowledge components.

        Returns:
            A tuple of trueskill Rating objects
            created from the given iterable of knowledge components.
        """
        return tuple(
            map(
                lambda kc: self._env.create_rating(
                    mu=kc.mean, sigma=math.sqrt(kc.variance)
                ),
                kcs,
            )
        )

    @abstractmethod
    def _update_knowledge_representation(self, x: EventModel, y: bool) -> None:
        """Update the knowledge representation of the LearnerModel.

        Args:
          x: A representation of the learning event.
          y: A bool indicating whether the learner engages in the learning event.
        """

    @final
    def fit(self, x: EventModel, y: bool) -> Self:
        # if positive_only is False or (it's true and y is true)
        # update the knowledge representation
        if not self.positive_only or y is True:
            self._update_knowledge_representation(x, y)

        self.__update_engagement_stats(y)
        self.__setup_env()
        return self

    @final
    def predict(self, x: EventModel) -> bool:
        return self.predict_proba(x) > self.threshold


def team_sum_quality(
    learner_kcs: Iterable[AbstractKnowledgeComponent],
    content_kcs: Iterable[AbstractKnowledgeComponent],
    beta: float,
) -> float:
    """Return the probability that the learner engages with the learnable unit.

    Args:
        learner_kcs: An iterable of knowledge components that come from the learner.
        content_kcs: An iterable of knowledge components that come from the content.
        beta: The noise factor.

    Returns:
        The probability that the learner engages with the learnable unit.
    """
    # make them list because we use them more than one time later
    learner_kcs = list(learner_kcs)
    content_kcs = list(content_kcs)

    team_learner_mean = map(lambda kc: kc.mean, learner_kcs)
    team_learner_variance = map(lambda kc: kc.variance, learner_kcs)
    team_content_mean = map(lambda kc: kc.mean, content_kcs)
    team_content_variance = map(lambda kc: kc.variance, content_kcs)

    difference = sum(team_learner_mean) - sum(team_content_mean)
    std = math.sqrt(sum(team_learner_variance) + sum(team_content_variance) + beta)
    return float(mpmath.ncdf(difference, mu=0, sigma=std))


def select_topic_kc_pairs(
    learner_model: LearnerModel,
    content_knowledge: Knowledge,
    init_skill: float,
    def_var: float,
    def_timestamp: Optional[float] = None,
) -> Iterable[Tuple[Hashable, AbstractKnowledgeComponent]]:
    """Get topic_id and knowledge_component pairs in the learner's knowledge \
    based on the knowledge of the learnable unit.

    Given the knowledge representation of the learnable unit, this method tries to get
    the corresponding knowledge representation from the Learner Model.

    If it cannot find the corresponding knowledge component in learner's model,
    which means the learner has never exposed to this knowledge component before,
    a new KC will be constructed with initial skill and default variance.

    Args:
        learner_model: A representation of the learner.
        content_knowledge: A representation of the knowledge of a learnable unit.
        init_skill: The initial mean of the newly created knowledge component.
        def_var: The initial variance of the newly created knowledge component.
        def_timestamp: The initial timestamp of the newly created knowledge component.
            If it's None, the newly created knowledge component has None as timestamp.

    Returns:
        An iterable of tuples consisting of (topic_id, knowledge_component) where
        topic_id is a hashable object that uniquely identifies a knowledge component,
        and the knowledge_component is the corresponding knowledge component of
        this topic_id.
    """

    def __topic_kc_pair_mapper(
        topic_kc_pair: Tuple[Hashable, AbstractKnowledgeComponent]
    ) -> Tuple[Hashable, AbstractKnowledgeComponent]:
        topic_id, kc = topic_kc_pair
        extracted_kc = learner_model.knowledge.get_kc(
            topic_id,
            kc.clone(mean=init_skill, variance=def_var, timestamp=def_timestamp),
        )
        return topic_id, extracted_kc

    team_learner = map(__topic_kc_pair_mapper, content_knowledge.topic_kc_pairs())
    return team_learner


def select_kcs(
    learner_model: LearnerModel,
    content_knowledge: Knowledge,
    init_skill: float,
    def_var: float,
    def_timestamp: Optional[float] = None,
) -> Iterable[AbstractKnowledgeComponent]:
    """Get knowledge components in the learner's knowledge \
    based on the knowledge of the learnable unit.

    Given the knowledge representation of the learnable unit, this method tries to get
    the corresponding knowledge representation from the Learner Model.

    If it cannot find the corresponding knowledge component in learner's model,
    which means the learner has never exposed to this knowledge component before,
    a new KC will be constructed with initial skill and default variance.

    Args:
        learner_model: A representation of the learner.
        content_knowledge: A representation of the knowledge of a learnable unit.
        init_skill: The initial mean of the newly created knowledge component.
        def_var: The initial variance of the newly created knowledge component.
        def_timestamp: The initial timestamp of the newly created knowledge component.
            If it's None, the newly created knowledge component has None as timestamp.

    Returns:
        An iterable of knowledge components.
    """

    def __kc_mapper(
        topic_kc_pair: Tuple[Hashable, AbstractKnowledgeComponent]
    ) -> AbstractKnowledgeComponent:
        topic_id, kc = topic_kc_pair
        extracted_kc = learner_model.knowledge.get_kc(
            topic_id,
            kc.clone(mean=init_skill, variance=def_var, timestamp=def_timestamp),
        )
        return extracted_kc

    team_learner = map(__kc_mapper, content_knowledge.topic_kc_pairs())
    return team_learner
