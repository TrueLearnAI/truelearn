# pylint: disable=missing-function-docstring
from pytest_socket import disable_socket, enable_socket

from truelearn import datasets


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


def test_load_peek_dataset_with_limit():
    train, test, mapping = datasets.load_peek_dataset(train_limit=1, test_limit=1)

    assert len(train) == 1
    assert len(test) == 1
    assert len(mapping) == 30367


def test_load_peek_dataset_def_var():
    train, *_ = datasets.load_peek_dataset(train_limit=1, test_limit=1, variance=0.5)
    _, events = train[0]

    kcs = events[0][0].knowledge.knowledge_components()
    for kc in kcs:
        assert kc.variance == 0.5


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


def test_load_peek_dataset_raw_with_limit():
    train, test, mapping = datasets.load_peek_dataset_raw(train_limit=1, test_limit=1)

    assert len(train) == 1
    assert len(test) == 1
    assert len(mapping) == 30367
