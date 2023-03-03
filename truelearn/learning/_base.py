import math
import collections
from abc import ABC, abstractmethod
from typing import Iterable, Hashable, Any, Optional, Tuple, Dict
from typing_extensions import Self, Final, final

import trueskill
import mpmath

from truelearn.models import (
    EventModel,
    AbstractKnowledgeComponent,
    LearnerModel,
)


class BaseClassifier(ABC):
    """The base class of all the classifiers in TrueLearn.

    Every subclass of the BaseClassifier should define their
    `_parameter_constraints`. This specified the parameters exposed
    via the `get_params` method and the constraints on the type of
    the parameters.

    The `_parameter_constraints` is a dictionary that maps parameter
    names to its expected type. The expected type can be a list or a single type
    or a tuple of values as it's possible for a type to accept more than one type/value.
    To do the constraint check based on this, simply call `self._validate_params()`
    in your classifier.
    """

    __DEEP_PARAM_DELIMITER: Final[str] = "__"

    # TODO: use constraint and satisfies to validate parameters
    # as it gives us more flexibility and can help us eliminate the
    # checks in InterestNoveltyKnowledgeBaseClassifier/INKClassifier
    # (see scikit-learn _base)
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
            if not hasattr(self, key):
                raise ValueError(
                    f"The specified parameter name {key}"
                    f" is not in the {type(self)}."
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
                    f"The given parameter {key}" f" is not in the {type(self)}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        # verify that the new parameters are valid
        self._validate_params()

        return self

    @final
    def _validate_params(self) -> None:
        """Validate types of given arguments in __init__.

        Args:
            **kargs: A dict of (param_name, param_value) pair.

        Raises:
            TypeError:
                types of parameters mismatch their constraints.
            ValueError:
                If the parameter is not any of the valid values in the given tuple.
        """
        for (
            param_name,
            expected_param_type,
        ) in self._parameter_constraints.items():
            # ignore constraints for non-existing attributes
            if param_name not in self.__dict__:
                continue

            param_value = self.__dict__[param_name]

            if isinstance(expected_param_type, list):
                # if it matches none of the types in the constraints
                if not any(
                    isinstance(self.__dict__[param_name], param_type_unpacked)
                    for param_type_unpacked in list(expected_param_type)
                ):
                    param_classname_expected = list(
                        map(lambda cls: cls.__name__, expected_param_type)
                    )
                    raise TypeError(
                        f"The {param_name} parameter of class {type(self)}"
                        f" __init__ function must be one of the classes"
                        f" in {param_classname_expected!r}."
                        f" Got {type(param_value)} instead."
                    )
            elif isinstance(expected_param_type, tuple):
                if param_value not in expected_param_type:
                    raise ValueError(
                        f"The {param_name} parameter of {type(self)}"
                        " must be one of the value inside "
                        f"tuple {expected_param_type!r}. Got {param_value!r} instead."
                    )
            else:
                if not isinstance(param_value, expected_param_type):
                    raise TypeError(
                        f"The {param_name} parameter of {type(self)}"
                        f" must be {expected_param_type!r}."
                        f" Got {type(param_value)} instead."
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

    __DEFAULT_GLOBAL_SIGMA: Final[float] = 1e-9
    __DEFAULT_DRAW_PROBA_LOW: Final[float] = 1e-9
    __DEFAULT_DRAW_PROBA_HIGH: Final[float] = 0.999999999

    _parameter_constraints: Dict[str, Any] = {
        **BaseClassifier._parameter_constraints,
        "learner_model": [LearnerModel, type(None)],
        "threshold": float,
        "init_skill": float,
        "def_var": float,
        "tau": float,
        "beta": float,
        "positive_only": bool,
        "draw_proba_type": ("static", "dynamic"),
        "draw_proba_static": [float, type(None)],
        "draw_proba_factor": float,
    }

    @staticmethod
    def _team_sum_quality(
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

    @staticmethod
    def _gather_trueskill_team(
        env: trueskill.TrueSkill, kcs: Iterable[AbstractKnowledgeComponent]
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

        self.draw_proba_type = draw_proba_type
        self.draw_proba_factor = draw_proba_factor
        self.draw_proba_static = draw_proba_static

        # check to ensure that the constructed classifier is not in corrupt state
        if self.draw_proba_type == "static" and self.draw_proba_static is None:
            raise ValueError(
                "When draw_proba_type is set to static,"
                " the draw_proba_static should not be None."
            )

    def __calculate_draw_proba(self) -> float:
        if self.draw_proba_type == "static":
            # delayed check as draw_proba_type can be potentially replaced by set_params
            # we can declare a version of constraint checker once we support
            # satisfies-based constraint checking
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
                InterestNoveltyKnowledgeBaseClassifier.__DEFAULT_DRAW_PROBA_HIGH,
                draw_probability,
            ),
            InterestNoveltyKnowledgeBaseClassifier.__DEFAULT_DRAW_PROBA_LOW,
        )

        # draw_proba_param is a factor if the type is dynamic
        return draw_probability * self.draw_proba_factor

    def __create_env(self) -> trueskill.TrueSkill:
        """Create the trueskill environment used in the training/prediction process."""
        draw_probability = self.__calculate_draw_proba()
        return trueskill.TrueSkill(
            mu=0.0,
            sigma=InterestNoveltyKnowledgeBaseClassifier.__DEFAULT_GLOBAL_SIGMA,
            beta=self.beta,
            tau=self.tau,
            draw_probability=draw_probability,
            backend="mpmath",
        )

    def __update_engagement_stats(self, y: bool) -> None:
        """Update the learner's engagement stats based on the given label.

        Args:
            y: A bool indicating whether the learner engage in the learning event.
        """
        if y:
            self.learner_model.number_of_engagements += 1
        else:
            self.learner_model.number_of_non_engagements += 1

    def __select_kcs(self, x: EventModel) -> Iterable[AbstractKnowledgeComponent]:
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
            topic_kc_pair: Tuple[Hashable, AbstractKnowledgeComponent]
        ) -> AbstractKnowledgeComponent:
            topic_id, kc = topic_kc_pair
            extracted_kc = self.learner_model.knowledge.get_kc(
                topic_id,
                kc.clone(
                    mean=self.init_skill,
                    variance=self.def_var,
                    timestamp=x.event_time,
                ),
            )
            return extracted_kc

        team_learner = map(__kc_mapper, x.knowledge.topic_kc_pairs())
        return team_learner

    def __select_topic_kc_pairs(
        self, x: EventModel
    ) -> Iterable[Tuple[Hashable, AbstractKnowledgeComponent]]:
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
            topic_kc_pair: Tuple[Hashable, AbstractKnowledgeComponent]
        ) -> Tuple[Hashable, AbstractKnowledgeComponent]:
            topic_id, kc = topic_kc_pair
            extracted_kc = self.learner_model.knowledge.get_kc(
                topic_id,
                kc.clone(
                    mean=self.init_skill, variance=self.def_var, timestamp=x.event_time
                ),
            )
            return topic_id, extracted_kc

        team_learner = map(__topic_kc_pair_mapper, x.knowledge.topic_kc_pairs())
        return team_learner

    @abstractmethod
    def _generate_ratings(  # pylint: disable=too-many-arguments
        self,
        env: trueskill.TrueSkill,
        learner_kcs: Iterable[AbstractKnowledgeComponent],
        content_kcs: Iterable[AbstractKnowledgeComponent],
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
            self.learner_model.knowledge.update_kc(topic_id, kc)

    @final
    def fit(self, x: EventModel, y: bool) -> Self:
        # if positive_only is False or (it's true and y is true)
        # update the knowledge representation
        if not self.positive_only or y is True:
            env = self.__create_env()
            self.__update_knowledge_representation(env, x, y)

        self.__update_engagement_stats(y)
        return self

    @final
    def predict(self, x: EventModel) -> bool:
        return self.predict_proba(x) > self.threshold

    @abstractmethod
    def _eval_matching_quality(
        self,
        env: trueskill.TrueSkill,
        learner_kcs: Iterable[AbstractKnowledgeComponent],
        content_kcs: Iterable[AbstractKnowledgeComponent],
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
        return self.learner_model
