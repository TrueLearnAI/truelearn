# pylint: disable=missing-function-docstring
from truelearn import preprocessing


def test_get_values_mean():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert preprocessing.get_values_mean(values) == 3.0


def test_get_values_population_std():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert preprocessing.get_values_population_std(values) == 1.4142135623730951


def test_get_values_sample_std():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert preprocessing.get_values_sample_std(values) == 1.5811388300841898
