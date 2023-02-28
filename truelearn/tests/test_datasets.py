# pylint: disable=missing-function-docstring
from truelearn import datasets

from pytest_socket import disable_socket, enable_socket


def test_load_peek_dataset():
    train, test, mapping = datasets.load_peek_dataset()

    assert len(train) == 14050
    assert len(test) == 5969
    assert len(mapping) == 30367


def test_load_peek_dataset_cache():
    train, test, mapping = datasets.load_peek_dataset()

    assert len(train) == 14050
    assert len(test) == 5969
    assert len(mapping) == 30367

    disable_socket()

    train, test, mapping = datasets.load_peek_dataset()

    assert len(train) == 14050
    assert len(test) == 5969
    assert len(mapping) == 30367

    enable_socket()


def test_load_peek_dataset_raw():
    train, test, mapping = datasets.load_peek_dataset_raw()

    assert len(train) == 203590
    assert len(test) == 86945
    assert len(mapping) == 30367


def test_load_peek_dataset_raw_cache():
    train, test, mapping = datasets.load_peek_dataset_raw()

    assert len(train) == 203590
    assert len(test) == 86945
    assert len(mapping) == 30367

    disable_socket()

    train, test, mapping = datasets.load_peek_dataset_raw()

    assert len(train) == 203590
    assert len(test) == 86945
    assert len(mapping) == 30367

    enable_socket()
