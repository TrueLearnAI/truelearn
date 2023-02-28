import statistics
from typing import Iterable


def get_values_mean(cosines: Iterable[float]) -> float:
    """Calculate the mean of an iterable of values.

    Args:
        cosines: An iterable of float values.

    Returns:
        The mean of the values in the iterable.
    """
    return statistics.mean(cosines)


def get_values_sample_std(cosines: Iterable[float]) -> float:
    """Calculate the sample standard deviation of an iterable of values.

    Args:
        cosines: An iterable of float values.

    Returns:
        The sample standard deviation of the values in the iterable.
    """
    return statistics.stdev(cosines)


def get_values_population_std(cosines: Iterable[float]) -> float:
    """Calculate the population standard deviation of an iterable of values.

    Args:
        cosines: An iterable of float values.

    Returns:
        The population standard deviation of the values in the iterable.
    """
    return statistics.pstdev(cosines)
