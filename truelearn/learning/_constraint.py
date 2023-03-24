from typing import Any, Callable

from ._base import BaseClassifier


class TypeConstraint:
    """Represent a tuple of type constraints.

    Examples:
        >>> # Init type constraints with many types
        >>> TypeConstraint(int, float, str)
        TypeConstraint(<class 'int'>, <class 'float'>, <class 'str'>)
    """

    def __init__(self, *type_constraints: type):
        """Init the TypeConstraint class.

        Args:
            *type_constraints: A tuple of type constraints
        """
        self.type_constraints = type_constraints

    def __repr__(self):
        """Get a description of the TypeConstraint class.

        Returns:
            A description of the TypeConstraint class.
        """
        return f"TypeConstraint{self.type_constraints}"

    def satisfies(self, obj: BaseClassifier, param_name: str):
        """Check if the given the value corresponds to the given param_name\
        satisfies the constraints.

        Args:
            obj: Any classifier.
            param_name: A str representing the name of the parameter.

        Raises:
            TypeError:
                if the given parameter doesn't match any of the types.
        """
        # pylint: disable=protected-access
        param_value = getattr(obj, obj._PARAM_PREFIX + param_name)

        if type(param_value) not in self.type_constraints:
            param_classname_expected = list(
                map(lambda cls: cls.__name__, self.type_constraints)
            )
            raise TypeError(
                f"The {param_name} parameter of class {obj.__class__.__name__!r}"
                " must be one of the classes"
                f" in {param_classname_expected!r}."
                f" Got {param_value.__class__.__name__!r} instead.",
            )


class ValueConstraint:
    """Represent a tuple of value constraints.

    Examples:
        >>> # Init value constraints with many types
        >>> ValueConstraint(1, 2, 3)
        ValueConstraint(1, 2, 3)
    """

    def __init__(self, *value_constraints: Any):
        """Init the ValueConstraint class.

        Args:
            *value_constraints: A tuple of value constraints
        """
        self.value_constraints = value_constraints

    def __repr__(self):
        """Get a description of the ValueConstraint class.

        Returns:
            A description of the ValueConstraint class.
        """
        return f"ValueConstraint{self.value_constraints}"

    def satisfies(self, obj: BaseClassifier, param_name: str):
        """Check if the given the value corresponds to the given param_name\
        satisfies the constraints.

        Args:
            obj: Any classifier.
            param_name: A str representing the name of the parameter.

        Raises:
            ValueError:
                if the given parameter doesn't match any of the values.
        """
        # pylint: disable=protected-access
        param_value = getattr(obj, obj._PARAM_PREFIX + param_name)

        if param_value not in self.value_constraints:
            raise ValueError(
                f"The {param_name} parameter of class {obj.__class__.__name__!r}"
                " must be one of the value inside "
                f"tuple {self.value_constraints!r}. Got {param_value!r} instead.",
            )


class FuncConstraint:
    """Represent a tuple of constraints based on functions.

    Examples:
        >>> # Init value constraints with many types
        >>> FuncConstraint(lambda x, y: None, lambda a, b: None)
        FuncConstraint(number_of_funcs=2)
    """

    def __init__(self, *func_constraints: Callable[[BaseClassifier, str], None]):
        """Init the FuncConstraint class.

        Args:
            func_constraints:
                A tuple of callable functions.
                The function should raise an error if
                any of the type constraints is violated.
        """
        self.func_constraints = func_constraints

    def __repr__(self):
        """Get a description of the FuncConstraint class.

        Returns:
            A description of the FuncConstraint class.
        """
        return f"FuncConstraint(number_of_funcs={len(self.func_constraints)})"

    def satisfies(self, obj: BaseClassifier, param_name: str):
        """Check if the given the value corresponds to the given param_name\
        satisfies the constraints.

        Args:
            obj: Any classifier.
            param_name: A str representing the name of the parameter.

        Raises:
            If the given parameter violates any of the type constraints.
        """
        for fn in self.func_constraints:
            fn(obj, param_name)
