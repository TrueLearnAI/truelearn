from typing import Any
from typing_extensions import Self
from abc import ABC, abstractmethod


class AbstractKnowledgeComponent(ABC):
    """An interface defines an knowledge component of a learnable unit.

    Each knowledge component can be represented as a Normal Distribution with certain skills (mu) and
    standard deviation (sqrt{variance}).

    The variance of the knowledge component from the learnable unit will be a fixed small value
    as we assume the skill (recourse depth) clearly and accurately measure the resourcefulness
    of the learnable unit. The variance of the knowledge component from the learner will be a dynamic
    value derived from the classifier's training process. This is to respect the fact that the skill
    (model's understanding of the learner's knowledge) is not perfectly accurate.

    Attributes:
        mean: A float indicating the mean of the knowledge component.
        variance: A float indicating the variance of the knowledge component.
        timestamp: A float indicating the POSIX timestamp of the last update of the knowledge component.
    """

    @property
    @abstractmethod
    def mean(self) -> float:
        """The mean of the knowledge component."""

    @property
    @abstractmethod
    def variance(self) -> float:
        """The variance of the knowledge component."""

    @property
    @abstractmethod
    def timestamp(self) -> float | None:
        """The POSIX timestamp of the last update of the knowledge component."""

    @abstractmethod
    def update(
        self,
        *,
        mean: float | None = None,
        variance: float | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Update the mean, variance, and timestamp of the current knowledge component.

        If the given parameters are None, the corresponding attributes of the current
        knowledge component will not be updated.

        Args:
          *: Use to reject positional arguments.
          mean: The new mean of the knowledge component.
          variance: The new variance of the knowledge component.
          timestamp: The new POSIX timestamp that indicates the update time of the knowledge component.
        """

    @abstractmethod
    def clone(
        self,
        *,
        mean: float,
        variance: float,
        timestamp: float | None = None,
    ) -> Self:
        """Generate a copy of the current knowledge component with given mean, variance and timestamp.

        Args:
            *: Use to reject positional arguments.
            mean: The new mean of the AbstractKnowledgeComponent.
            variance: The new variance of the AbstractKnowledgeComponent.
            timestamp: An optional new POSIX timestamp of the AbstractKnowledgeComponent.

        Returns:
            A cloned knowledge component with given mean, variance and timestamp.
        """

    @abstractmethod
    def export(self, output_format: str) -> Any:
        """Export the AbstractKnowledgeComponent into some formats.

        Args:
            output_format: The name of the output format.

        Returns:
            The requested format.

        Raises:
            ValueError: An unsupported format is given.
        """
