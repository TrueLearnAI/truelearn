"""The truelearn.datasets module contains utilities to load datasets, \
such as the PEEK dataset."""

from ._base import (
    check_and_download_file,
    RemoteFileMetaData,
)

from ._peek import (
    load_peek_dataset,
    load_peek_dataset_raw,
    PEEKData,
    PEEKKnowledgeComponentGenerator,
)

__all__ = [
    "check_and_download_file",
    "load_peek_dataset",
    "load_peek_dataset_raw",
    "PEEKData",
    "PEEKKnowledgeComponentGenerator",
    "RemoteFileMetaData",
]
