import collections
from abc import ABC, abstractmethod
from typing import Dict, Any
from typing_extensions import Self, Final, final, Literal

from truelearn.errors import InvalidArgumentError
from truelearn.models import EventModel


__all__ = ["BaseClassifier"]


class BaseClassifier(ABC):
    """The base class of all the classifiers in TrueLearn."""

    _PARAM_PREFIX: str = "_"

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
                contained sub-objects that inherit BaseClassifier class.

        Returns:
            A dict mapping variable names to the corresponding objects.
        """
        param_names = list(self._parameter_constraints.keys())
        param_names.sort()

        out = {}
        for key in param_names:
            if not hasattr(self, self._PARAM_PREFIX + key):
                continue

            value = getattr(self, self._PARAM_PREFIX + key)
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
            TrueLearnTypeError:
                Types of parameters do not satisfy their constraints.
            TrueLearnValueError:
                Values of parameters do not satisfy their constraints.
            InvalidArgumentError:
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
                raise InvalidArgumentError(
                    f"The given argument {key}"
                    f" is not in the class {self.__class__.__name__!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, self._PARAM_PREFIX + key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        # verify that the new parameters are valid
        self._validate_params()

        return self

    @final
    def _validate_params(self) -> None:
        """Validate types of given arguments in __init__.

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

        Raises:
            TrueLearnTypeError:
                Types of parameters do not satisfy their constraints.
            TrueLearnValueError:
                Values of parameters do not satisfy their constraints.
        """
        for (
            param_name,
            param_constraint,
        ) in self._parameter_constraints.items():
            if not hasattr(self, self._PARAM_PREFIX + param_name):
                continue
            if isinstance(param_constraint, list):
                for constraint in param_constraint:
                    constraint.satisfies(self, param_name)
            else:
                param_constraint.satisfies(self, param_name)
