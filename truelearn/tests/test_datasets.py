# pylint: disable=missing-function-docstring,missing-class-docstring,protected-access
from urllib import request

import pytest
from pytest_socket import disable_socket, enable_socket

from truelearn import datasets
import truelearn.datasets.base as base


class TestBase:
    @pytest.mark.disable_socket
    def test_download_non_https_url(self):
        with pytest.raises(ValueError) as excinfo:
            base._download_file(
                filepath=".",
                url="http://",
                expected_sha256="should not reach this",
                verbose=True,
            )
        assert "The given url http:// is not a valid https url." == str(excinfo.value)

    @pytest.mark.disable_socket
    def test_verbose_mode(self, capsys, monkeypatch):
        def mock_urlretrieve(*_args, **_kwargs):
            ...

        def mock_sha256sum(filepath):
            return filepath

        monkeypatch.setattr(request, "urlretrieve", mock_urlretrieve)
        monkeypatch.setattr(base, "_sha256sum", mock_sha256sum)

        base._download_file(
            filepath=".",
            url="https://",
            expected_sha256=".",
            verbose=True,
        )
        captured = capsys.readouterr()
        assert "Downloading https:// into .\n" == captured.out

    @pytest.mark.disable_socket
    def test_sha256sum(self, tmp_path, monkeypatch):
        def mock_urlretrieve(*_args, **_kwargs):
            ...

        monkeypatch.setattr(request, "urlretrieve", mock_urlretrieve)

        directory = tmp_path
        filepath = directory / "truelearn_sha256sum_test.txt"
        filepath.write_text("")

        with pytest.raises(IOError) as excinfo:
            base._download_file(
                filepath=str(filepath),
                url="https://",
                expected_sha256="1",
                verbose=True,
            )
        assert (
            "truelearn_sha256sum_test.txt has an SHA256 checksum "
            "(e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855) "
            "differing from expected (1), file may be corrupted." in str(excinfo.value)
        )


class TestPEEKDataset:
    def test_load_peek_dataset(self):
        train, test, mapping = datasets.load_peek_dataset()

        assert len(train) == 14050
        assert len(test) == 5969
        assert len(mapping) == 30367

    def test_load_peek_dataset_cache(self):
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

    def test_load_peek_dataset_with_limit(self):
        train, test, mapping = datasets.load_peek_dataset(train_limit=1, test_limit=1)

        assert len(train) == 1
        assert len(test) == 1
        assert len(mapping) == 30367

    def test_load_peek_dataset_with_invalid_limit(self):
        with pytest.raises(ValueError) as excinfo:
            datasets.load_peek_dataset(train_limit=-1)
        assert "train_limit must >= 0. Got train_limit=-1 instead." == str(
            excinfo.value
        )

        with pytest.raises(ValueError) as excinfo:
            datasets.load_peek_dataset(test_limit=-1)
        assert "test_limit must >= 0. Got test_limit=-1 instead." == str(excinfo.value)

    def test_load_peek_dataset_def_var(self):
        train, *_ = datasets.load_peek_dataset(
            train_limit=1, test_limit=1, variance=0.5
        )
        _, events = train[0]

        kcs = events[0][0].knowledge.knowledge_components()
        for kc in kcs:
            assert kc.variance == 0.5

    def test_load_peek_dataset_raw(self):
        train, test, mapping = datasets.load_peek_dataset_raw()

        assert len(train) == 203590
        assert len(test) == 86945
        assert len(mapping) == 30367

    def test_load_peek_dataset_raw_cache(self):
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

    def test_load_peek_dataset_raw_with_limit(self):
        train, test, mapping = datasets.load_peek_dataset_raw(
            train_limit=1, test_limit=1
        )

        assert len(train) == 1
        assert len(test) == 1
        assert len(mapping) == 30367

    def test_load_peek_dataset_raw_with_invalid_limit(self):
        with pytest.raises(ValueError) as excinfo:
            datasets.load_peek_dataset_raw(train_limit=-1)
        assert "train_limit must >= 0. Got train_limit=-1 instead." == str(
            excinfo.value
        )

        with pytest.raises(ValueError) as excinfo:
            datasets.load_peek_dataset_raw(test_limit=-1)
        assert "test_limit must >= 0. Got test_limit=-1 instead." == str(excinfo.value)
