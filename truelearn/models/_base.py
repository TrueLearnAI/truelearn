from abc import abstractmethod
from typing import Any, Optional, Dict
from typing_extensions import Self, Protocol


class BaseKnowledgeComponent(Protocol):
    """An interface defines a knowledge component of a learnable unit.

    Each knowledge component can be represented as a Normal Distribution with
    certain skills (mu) and standard deviation (sqrt{variance}).

    The variance of the knowledge component from the learnable unit will be
    a fixed small value as we assume the skill (recourse depth) clearly and
    accurately measure the resourcefulness of the learnable unit.
    The variance of the knowledge component from the learner will be a dynamic
    value derived from the classifier's training process. This is to respect the
    fact that the skill (model's understanding of the learner's knowledge) is
    not perfectly accurate.
    """

    @property
    @abstractmethod
    def mean(self) -> float:
        """float: The mean of the knowledge component."""

    @property
    @abstractmethod
    def variance(self) -> float:
        """float: The variance of the knowledge component."""

    @property
    @abstractmethod
    def timestamp(self) -> Optional[float]:
        """Optional[float]: The POSIX timestamp of when the knowledge component was \
        last updated."""

    @abstractmethod
    def update(
        self,
        *,
        mean: Optional[float] = None,
        variance: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Update the mean, variance, and timestamp of the current knowledge component.

        If the given parameters are None, the corresponding attributes of the current
        knowledge component will not be updated.

        Args:
            *:
                Use to reject positional arguments.
            mean:
                The new mean of the knowledge component.
            variance:
                The new variance of the knowledge component.
            timestamp:
                The new POSIX timestamp that indicates the update time
                of the knowledge component.

        Returns:
            None.
        """

    @abstractmethod
    def clone(
        self,
        *,
        mean: float,
        variance: float,
        timestamp: Optional[float] = None,
    ) -> Self:
        """Generate a copy of the current knowledge component with \
        given mean, variance and timestamp.

        Args:
            *:
                Use to reject positional arguments.
            mean:
                The new mean of the knowledge component.
            variance:
                The new variance of the knowledge component.
            timestamp:
                An optional new POSIX timestamp of the knowledge component.

        Returns:
            A cloned knowledge component with given mean, variance and timestamp.
        """

    @abstractmethod
    def export_as_dict(self) -> Dict[str, Any]:
        """Export the knowledge component into a dictionary.

        Returns:
            A dictionary mapping the name of the variables to
            their value.
        """
