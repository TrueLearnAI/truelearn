import dataclasses
import hashlib
from typing import Optional
from urllib import request
from os import path


# pylint: disable=pointless-string-statement
"""
Copyright of `RemoteFileMetaData`, `download_file` are held by
[BSD 3-Clause License, scikit-learn developers, 2007-2022].
"""


@dataclasses.dataclass
class RemoteFileMetaData:
    """Remote file metadata.

    Args:
        url: The url of the file.
        filename: The filename of the downloaded file.
        sha256_expected: The expected sha256 sum of the file.
    """

    url: str
    filename: str
    sha256: str


def _sha256sum(filepath) -> str:
    buf_size = 65536
    h = hashlib.sha256()
    with open(filepath, "rb", buffering=0) as file:
        while True:
            chunk = file.read(buf_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _download_file(*, filepath: str, url: str, sha256sum: str) -> str:
    """Download a remote file and check the sha256.

    Args:
        remote_file:
            Some metadata about the remote file.
        dirname:
            An optional path that specifies the location of the downloaded file.

    Returns:
        Full path of the created file.
    """
    if url.lower().startswith("http"):
        print(f"Downloading {url} into {filepath}")
        # the bandit warning is suppressed here
        # because we have checked whether the url starts with http
        request.urlretrieve(url, filepath)  # nosec
    else:
        raise ValueError(f"The given url {url} is not a valid http/https url.")

    sha256_actual = _sha256sum(filepath)
    if sha256sum != sha256_actual:
        raise IOError(
            f"{filepath} has an SHA256 checksum ({sha256_actual}) "
            f"differing from expected ({sha256sum}), "
            "file may be corrupted."
        )

    return filepath


def check_and_download_file(
    *, remote_file: RemoteFileMetaData, dirname: Optional[str] = None
) -> str:
    """Download a remote file and check the sha256.

    If the file already exists and matches the given sha256 sum value,
    the function will not download the file again.

    Args:
        remote_file:
            Some metadata about the remote file.
        dirname:
            An optional path that specifies the location of the downloaded file.

    Returns:
        Full path of the created file.
    """
    filepath = (
        remote_file.filename
        if dirname is None
        else path.join(dirname, remote_file.filename)
    )

    # if already downloaded
    if (
        path.exists(filepath)
        and path.isfile(filepath)
        and remote_file.sha256 == _sha256sum(filepath)
    ):
        return filepath

    return _download_file(
        filepath=filepath, url=remote_file.url, sha256sum=remote_file.sha256
    )
