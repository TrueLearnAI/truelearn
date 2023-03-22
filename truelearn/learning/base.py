import collections
import math
from abc import ABC, abstractmethod
from typing import Iterable, Hashable, Optional, Tuple, Dict, Any
from typing_extensions import Self, Final, final, Literal

import trueskill
import mpmath

from truelearn.models import (
    EventModel,
    BaseKnowledgeComponent,
    LearnerModel,
)
from ._constraint import (
    TypeConstraint,
    ValueConstraint,
    FuncConstraint,
    validate_params,
)


class BaseClassifier(ABC):
    """The base class of all the classifiers in TrueLearn.

    Every subclass of the BaseClassifier should define their
    `_parameter_constraints`. This specified the parameters exposed
    via the `get_params` method and the constraints on the type of
    the parameters.

    The `_parameter_constraints` is a dictionary that maps parameter
    names to its constraint/list of constraints.
    The constraint types are defined in ._constraints. Each constraint
    type defines its way of determining whether the parameter `satisfies`
    the constraint.
    Having a list of constraints as the value in the dictionary means that
    all of them must be satisfied. Notice that the order of the constraints
    in the list is important as the `self._validate_params` validates them in
    sequential order.
    To do the constraint check based on this, simply call `self._validate_params()`
    in your classifier.

    Attributes:
        PARAM_PREFIX:
            The prefix of all the instance attributes in the classifier.
            This is to help get_params/set_params/_validate_params work correctly.
            The reason why we choose to append a prefix "_" by default to all variables
            is that we don't want the user feel like they can directly access the
            instance attributes of the classifier. In fact, any direct read/write to the
            instance attributes are not encouraged as they may lead to unexpected
            behaviour. The recommended way to do this is to read variables via
            `get_params` and set variables via `set_params`. As these are not
            common operations, the performance impact is negligible.
            Implementations of Classifiers can override this class attribute. But,
            it's not recommended to do that. The recommendation is to prefix all
            instance attributes with "_" as this is consistent with the current
            implementation.
    """

    PARAM_PREFIX: str = "_"

    __DEEP_PARAM_DELIMITER: Final[Literal["__"]] = "__"

    _parameter_constraints: Dict[str, Any] = {}

    def __repr__(self) -> str:
        """Get a description of the classifier object.

        Returns:
            A string description of the classifier object.
            Currently, this method only prints the name
            of the classifier object.
        """
        return f"{self.__class__.__name__}()"

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
            if not hasattr(self, self.PARAM_PREFIX + key):
                raise ValueError(
                    f"The specified parameter name {key}"
                    f" is not in the class {self.__class__.__name__!r}."
                )

            value = getattr(self, self.PARAM_PREFIX + key)
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
            KeyError:
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
                raise KeyError(
                    f"The given parameter {key}"
                    f" is not in the class {self.__class__.__name__!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, self.PARAM_PREFIX + key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        # verify that the new parameters are valid
        self._validate_params()

        return self

    @final
    def _validate_params(self) -> None:
        """Validate types of given arguments in __init__.

        Raises:
            TypeError:
                Types of parameters mismatch their constraints.
            ValueError:
                If the parameter is not any of the valid values in the given tuple.
        """
        validate_params(self, self._parameter_constraints)


def draw_proba_static_constraint(obj: BaseClassifier, _):
    """Check whether the draw_proba_static is properly set when\
    draw_proba_type is "static".

    Args:
        obj: The object to check.
        _: Use to accept param_name.

    Returns:
        a ValueError object if the draw_proba_static is not property set,
        which means it's None while draw_proba_type is static. If everything
        is properly set, it returns None.
    """
    params = obj.get_params(deep=False)
    if params["draw_proba_type"] == "static" and params["draw_proba_static"] is None:
        raise ValueError(
            "When draw_proba_type is set to static,"
            " the draw_proba_static should not be None."
        )


def team_sum_quality(
    *,
    learner_mean: Iterable[float],
    learner_variance: Iterable[float],
    content_mean: Iterable[float],
    content_variance: Iterable[float],
    beta: float,
) -> float:
    """Return the probability that the learner engages with the content.

    Args:
        learner_mean:
            An iterable of the mean of knowledge components that come from the learner.
        learner_variance:
            An iterable of the variance of knowledge components that come from
            the learner.
        content_mean:
            An iterable of the mean of knowledge components that come from the content.
        content_variance:
            An iterable of the variance of knowledge components that come from
            the content.
        beta:
            The distance which guarantees about 76% chance of winning.
            The recommended value is sqrt(def_var) / 2.

    Returns:
        The probability that the learner engages with the content.
    """
    difference = sum(learner_mean) - sum(content_mean)
    std = math.sqrt(sum(learner_variance) + sum(content_variance) + beta)
    return float(mpmath.ncdf(difference, mu=0, sigma=std))


def team_sum_quality_from_kcs(
    learner_kcs: Iterable[BaseKnowledgeComponent],
    content_kcs: Iterable[BaseKnowledgeComponent],
    beta: float,
) -> float:
    """Return the probability that the learner engages with the learnable unit.

    Args:
        learner_kcs:
            An iterable of knowledge components that come from the learner.
        content_kcs:
            An iterable of knowledge components that come from the content.
        beta:
            The distance which guarantees about 76% chance of winning.
            The recommended value is sqrt(def_var) / 2.

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

    return team_sum_quality(
        learner_mean=team_learner_mean,
        learner_variance=team_learner_variance,
        content_mean=team_content_mean,
        content_variance=team_content_variance,
        beta=beta,
    )


def gather_trueskill_team(
    env: trueskill.TrueSkill, kcs: Iterable[BaseKnowledgeComponent]
) -> Tuple[trueskill.Rating, ...]:
    """Return a tuple of trueskill Rating \
    created from the given iterable of knowledge components.

    Args:
        env: The trueskill environment where the training/prediction happens.
        kcs: An iterable of knowledge components.

    Returns:
        A tuple of trueskill Rating objects
        created from the given iterable of knowledge components.
    """
    return tuple(
        map(
            lambda kc: env.create_rating(mu=kc.mean, sigma=math.sqrt(kc.variance)),
            kcs,
        )
    )


class InterestNoveltyKnowledgeBaseClassifier(BaseClassifier):
    """A Base Classifier for KnowledgeClassifier, NoveltyClassifier \
    and InterestClassifier.

    It defines the necessary instance variables and
    common methods to interact with the LearnerModel.

    All the classifiers that inherit this class should
    define their own `__init__`, `_generate_ratings` and
    `_eval_matching_quality` methods.
    """

    __DEFAULT_DRAW_PROBA_LOW: Final[float] = 1e-9
    __DEFAULT_DRAW_PROBA_HIGH: Final[float] = 0.999999999

    _parameter_constraints: Dict[str, Any] = {
        **BaseClassifier._parameter_constraints,
        "learner_model": TypeConstraint(LearnerModel),
        "threshold": TypeConstraint(float),
        "init_skill": TypeConstraint(float),
        "def_var": TypeConstraint(float),
        "tau": TypeConstraint(float),
        "beta": TypeConstraint(float),
        "positive_only": TypeConstraint(bool),
        "draw_proba_type": ValueConstraint("static", "dynamic"),
        "draw_proba_static": [
            TypeConstraint(float, type(None)),
            FuncConstraint(draw_proba_static_constraint),
        ],
        "draw_proba_factor": TypeConstraint(float),
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
                The distance which guarantees about 76% chance of winning.
                The recommended value is sqrt(def_var) / 2.
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
            self._learner_model = LearnerModel()
        else:
            self._learner_model = learner_model
        self._threshold = threshold
        self._init_skill = init_skill
        self._def_var = def_var
        self._tau = tau
        self._beta = beta
        self._positive_only = positive_only

        self._draw_proba_type = draw_proba_type
        self._draw_proba_factor = draw_proba_factor
        self._draw_proba_static = draw_proba_static

    def __calculate_draw_proba(self) -> float:
        if self._draw_proba_type == "static":
            # we are sure that draw_proba_static cannot be None by using our type check
            return self._draw_proba_static * self._draw_proba_factor  # type: ignore

        # >= 1 because it's divisor
        total_engagement_stats = max(
            1,
            self._learner_model.number_of_engagements
            + self._learner_model.number_of_non_engagements,
        )
        draw_probability = float(
            self._learner_model.number_of_engagements / total_engagement_stats
        )

        # clamp the value between [DEFAULT_DRAW_PROBA_LOW, DEFAULT_DRAW_PROBA_HIGH]
        draw_probability = max(
            min(
                InterestNoveltyKnowledgeBaseClassifier.__DEFAULT_DRAW_PROBA_HIGH,
                draw_probability,
            ),
            InterestNoveltyKnowledgeBaseClassifier.__DEFAULT_DRAW_PROBA_LOW,
        )

        return draw_probability * self._draw_proba_factor

    def __create_env(self) -> trueskill.TrueSkill:
        """Create the trueskill environment used in the training/prediction process."""
        draw_probability = self.__calculate_draw_proba()
        return trueskill.TrueSkill(
            mu=self._init_skill,
            sigma=math.sqrt(self._def_var),
            beta=self._beta,
            tau=self._tau,
            draw_probability=draw_probability,
            backend="mpmath",
        )

    def __update_engagement_stats(self, y: bool) -> None:
        """Update the learner's engagement stats based on the given label.

        Args:
            y: A bool indicating whether the learner engage in the learning event.
        """
        if y:
            self._learner_model.number_of_engagements += 1
        else:
            self._learner_model.number_of_non_engagements += 1

    def __select_kcs(self, x: EventModel) -> Iterable[BaseKnowledgeComponent]:
        """Get knowledge components in the learner's knowledge \
        based on the knowledge of the learnable unit.

        Given the knowledge representation of the learnable unit,
        this method tries to get the corresponding knowledge representation
        from the Learner Model.

        If it cannot find the corresponding knowledge component in learner's model,
        which means the learner has never exposed to this knowledge component before,
        a new KC will be constructed with initial skill and default variance.

        Args:
            x: A representation of a learning event.

        Returns:
            An iterable of knowledge components.
        """

        def __kc_mapper(
            topic_kc_pair: Tuple[Hashable, BaseKnowledgeComponent]
        ) -> BaseKnowledgeComponent:
            topic_id, kc = topic_kc_pair
            extracted_kc = self._learner_model.knowledge.get_kc(
                topic_id,
                kc.clone(
                    mean=self._init_skill,
                    variance=self._def_var,
                    timestamp=x.event_time,
                ),
            )
            return extracted_kc

        team_learner = map(__kc_mapper, x.knowledge.topic_kc_pairs())
        return team_learner

    def __select_topic_kc_pairs(
        self, x: EventModel
    ) -> Iterable[Tuple[Hashable, BaseKnowledgeComponent]]:
        """Get topic_id and knowledge_component pairs in the learner's knowledge \
        based on the knowledge of the learnable unit.

        Given the knowledge representation of the learnable unit,
        this method tries to get the corresponding knowledge representation
        from the Learner Model.

        If it cannot find the corresponding knowledge component in learner's model,
        which means the learner has never exposed to this knowledge component before,
        a new KC will be constructed with initial skill and default variance.

        Args:
            x: A representation of a learning event.

        Returns:
            An iterable of tuples consisting of (topic_id, knowledge_component) where
            topic_id is a hashable object that uniquely identifies
            a knowledge component, and the knowledge_component is the corresponding
            knowledge component of this topic_id.
        """

        def __topic_kc_pair_mapper(
            topic_kc_pair: Tuple[Hashable, BaseKnowledgeComponent]
        ) -> Tuple[Hashable, BaseKnowledgeComponent]:
            topic_id, kc = topic_kc_pair
            extracted_kc = self._learner_model.knowledge.get_kc(
                topic_id,
                kc.clone(
                    mean=self._init_skill,
                    variance=self._def_var,
                    timestamp=x.event_time,
                ),
            )
            return topic_id, extracted_kc

        team_learner = map(__topic_kc_pair_mapper, x.knowledge.topic_kc_pairs())
        return team_learner

    @abstractmethod
    def _generate_ratings(  # pylint: disable=too-many-arguments
        self,
        env: trueskill.TrueSkill,
        learner_kcs: Iterable[BaseKnowledgeComponent],
        content_kcs: Iterable[BaseKnowledgeComponent],
        event_time: Optional[float],
        y: bool,
    ) -> Iterable[trueskill.Rating]:
        """Generate an iterable of the updated Rating for the learner.

        The Rating is generated based on the label and optionally
        event_time (for InterestClassifier).

        Args:
            env: The trueskill environment where the training/prediction happens.
            learner_kcs:
                An iterable of learner's knowledge components.
            content_kcs:
                An iterable of content's knowledge components.
            event_time:
                An optional float representing the event time.
            y:
                A bool indicating whether the learner engage in
                the learning event.

        Returns:
            An iterable of trueskill.Rating.
        """

    def __update_knowledge_representation(
        self, env: trueskill.TrueSkill, x: EventModel, y: bool
    ) -> None:
        """Update the knowledge representation of the LearnerModel.

        Args:
            env: The trueskill environment where the training/prediction happens.
            x: A representation of the learning event.
            y: A bool indicating whether the learner engages in the learning event.
        """
        # make it a list because it's used multiple times
        learner_topic_kc_pairs = list(
            self.__select_topic_kc_pairs(
                x,
            )
        )

        # extract learner kc from the topic_kc pairs above
        learner_kcs = map(
            lambda learner_topic_kc_pair: learner_topic_kc_pair[1],
            learner_topic_kc_pairs,
        )

        content_kcs = x.knowledge.knowledge_components()

        for topic_kc_pair, rating in zip(
            learner_topic_kc_pairs,
            self._generate_ratings(env, learner_kcs, content_kcs, x.event_time, y),
        ):
            topic_id, kc = topic_kc_pair
            kc.update(
                mean=rating.mu, variance=rating.sigma**2, timestamp=x.event_time
            )
            self._learner_model.knowledge.update_kc(topic_id, kc)

    @final
    def fit(self, x: EventModel, y: bool) -> Self:
        # if positive_only is False or (it's true and y is true)
        # update the knowledge representation
        if not self._positive_only or y:
            env = self.__create_env()
            self.__update_knowledge_representation(env, x, y)

        self.__update_engagement_stats(y)
        return self

    @final
    def predict(self, x: EventModel) -> bool:
        return self.predict_proba(x) > self._threshold

    @abstractmethod
    def _eval_matching_quality(
        self,
        env: trueskill.TrueSkill,
        learner_kcs: Iterable[BaseKnowledgeComponent],
        content_kcs: Iterable[BaseKnowledgeComponent],
    ) -> float:
        """Evaluate the matching quality of learner and content.

        Args:
            env:
                The trueskill environment where the training/prediction happens.
            learner_kcs:
                An iterable of learner's knowledge components.
            content_kcs:
                An iterable of content's knowledge components.

        Returns:
            A float between [0, 1], indicating the matching quality
            of the learner and the content. The higher the value,
            the better the match.
        """

    @final
    def predict_proba(self, x: EventModel) -> float:
        env = self.__create_env()
        learner_kcs = self.__select_kcs(x)
        content_kcs = x.knowledge.knowledge_components()

        return self._eval_matching_quality(env, learner_kcs, content_kcs)

    @final
    def get_learner_model(self) -> LearnerModel:
        """Get the learner model associated with this classifier.

        Returns:
            A learner model associated with this classifier.
        """
        return self._learner_model
