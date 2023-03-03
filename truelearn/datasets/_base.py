import dataclasses
import hashlib
from typing import Optional
from urllib import request
from os import path


@dataclasses.dataclass
class RemoteFileMetaData:
    """Remote file metadata.

    Args:
        url: The url of the file.
        filename: The filename of the downloaded file.
        expected_sha256: The expected sha256 sum of the file.
    """

    url: str
    filename: str
    expected_sha256: str


def _sha256sum(filepath) -> str:
    chunk_size = 65536
    hash_val = hashlib.sha256()
    with open(filepath, "rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            hash_val.update(chunk)
    return hash_val.hexdigest()


def _download_file(
    *, filepath: str, url: str, expected_sha256: str, verbose: bool
) -> None:
    """Download a remote file and check the sha256.

    Args:
        filepath:
            The full path of the created file.
        url:
            The url of the file.
        expected_sha256:
            The expected sha256 sum of the file.
        verbose:
            If True, this function outputs some information
            about the downloaded file.
    """
    if not url.lower().startswith("https://"):
        raise ValueError(f"The given url {url} is not a valid https url.")

    if verbose:
        print(f"Downloading {url} into {filepath}")
        # the bandit warning is suppressed here
        # because we have checked whether the url starts with http
        request.urlretrieve(url, filepath)  # nosec

    actual_sha256 = _sha256sum(filepath)
    if expected_sha256 != actual_sha256:
        raise IOError(
            f"{filepath} has an SHA256 checksum ({actual_sha256}) "
            f"differing from expected ({expected_sha256}), "
            "file may be corrupted."
        )


def check_and_download_file(
    *, remote_file: RemoteFileMetaData, dirname: Optional[str] = None, verbose: bool
) -> str:
    """Download a remote file and check the sha256.

    If the file already exists and matches the given sha256 sum value,
    the function will not download the file again.

    Args:
        remote_file:
            Some metadata about the remote file.
        dirname:
            An optional path that specifies the location of the downloaded file.
        verbose:
            If True and the downloaded file doesn't exist, this function outputs some
            information about the downloaded file.

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
        and remote_file.expected_sha256 == _sha256sum(filepath)
    ):
        return filepath

    _download_file(
        filepath=filepath,
        url=remote_file.url,
        expected_sha256=remote_file.expected_sha256,
        verbose=verbose,
    )

    return filepath
