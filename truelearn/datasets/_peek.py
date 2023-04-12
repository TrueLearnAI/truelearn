"""This module downloads data from PEEK-Dataset.

PEEK-Dataset is a Large Dataset of Learner Engagement with Educational Videos.
It contains data of a mapping from wikipedia id to wikipedia url, and events
separated into train and test data.

The columns of PEEK-Dataset mapping look like this: (id, url).

The columns of PEEK-Dataset train and test data look like this:
(slug, vid_id, part, time, timeframe, session, --topics--, label).
"""
import csv
import collections
import itertools
from typing import Tuple, Dict, List, Optional
from typing_extensions import TypeAlias, Protocol

from truelearn.errors import TrueLearnValueError
from truelearn.models import (
    EventModel,
    Knowledge,
    KnowledgeComponent,
    BaseKnowledgeComponent,
)
from ._base import RemoteFileMetaData, check_and_download_file


PEEKData: TypeAlias = List[Tuple[int, List[Tuple[EventModel, bool]]]]


PEEK_TRAIN = RemoteFileMetaData(
    url="https://raw.githubusercontent.com/sahanbull/PEEK-Dataset"
    "/1740aa04aeb019494fd3593d4f708fd519fa101a/datasets/v1/train.csv",
    filename="peek.train.csv",
    expected_sha256="291afa33dacec5b1f9751788cf4d2ba38cb495936f20d2edcda5c88a6e2539c8",
)

PEEK_TEST = RemoteFileMetaData(
    url="https://raw.githubusercontent.com/sahanbull/PEEK-Dataset"
    "/1740aa04aeb019494fd3593d4f708fd519fa101a/datasets/v1/test.csv",
    filename="peek.test.csv",
    expected_sha256="315d267fad18dcdf2300ede2fc0877c71cfc11c983a788ecf89bf69e3b4d2129",
)

PEEK_MAPPING = RemoteFileMetaData(
    url="https://raw.githubusercontent.com/sahanbull/PEEK-Dataset"
    "/213eee0ef6837468d12971a0d865de328ce69159/datasets/v2/"
    "id_to_wiki_metadata_mapping.csv",
    filename="peek.mapping.csv",
    expected_sha256="3240ac45661e8eefaaf120158f44a9aceba2e53117b07417ee5d3c055bb2d6c6",
)


class PEEKKnowledgeComponentGenerator(Protocol):
    """A class that defines an implicit interface\
    to the generator of the knowledge component."""

    def __call__(
        self,
        *,
        mean: float,
        variance: float,
        timestamp: float,
        url: str,
        title: str,
        description: str,
    ) -> BaseKnowledgeComponent:  # type: ignore
        """Generate a knowledge component.

        Args:
            *:
                Used to reject positional arguments.
            mean:
                The mean of the knowledge component.
            variance:
                The variance of the knowledge component.
            timestamp:
                The timestamp of the knowledge component.
            url:
                The url of the knowledge component.
            title:
                The title of the knowledge component.
            description:
                The description of the knowledge component.
        """


def __sanity_check(train_limit: Optional[int], test_limit: Optional[int]):
    """Check if train_limit and test_limit are valid.

    Args:
        train_limit:
            An optional non-negative integer specifying the
            maximum number of lines to read from the train file.
            If None, it means no limit.
        test_limit:
            An optional non-negative integer specifying the
            maximum number of lines to read from the test file.
            If None, it means no limit.

    Raises:
        TrueLearnValueError:
            If the train_limit or test_limit is less than 0.
    """
    if train_limit is not None and train_limit < 0:
        raise TrueLearnValueError(
            f"train_limit must >= 0. Got train_limit={train_limit} instead."
        )
    if test_limit is not None and test_limit < 0:
        raise TrueLearnValueError(
            f"test_limit must >= 0. Got test_limit={test_limit} instead."
        )


def __download_files(dirname: str = ".", verbose: bool = True) -> Tuple[str, str, str]:
    """Download PEEKDataset Files.

    Args:
        dirname:
            The destination directory of the downloaded file.
        verbose:
            Whether download in verbose mode.
            Verbose mode produces info about the downloaded files.

    Returns:
        A tuple of (train_filepath, test_filepath, mapping_filepath).
    """
    train_filepath = check_and_download_file(
        remote_file=PEEK_TRAIN, dirname=dirname, verbose=verbose
    )
    test_filepath = check_and_download_file(
        remote_file=PEEK_TEST, dirname=dirname, verbose=verbose
    )
    mapping_filepath = check_and_download_file(
        remote_file=PEEK_MAPPING, dirname=dirname, verbose=verbose
    )

    return train_filepath, test_filepath, mapping_filepath


def __restructure_line(
    line: List[str],
    id_to_url_mapping: Dict[int, Tuple[str, str, str]],
    variance: float,
    kc_init_func: PEEKKnowledgeComponentGenerator,
) -> Tuple[int, EventModel, bool]:
    """Restructure a single line in the file.

    Args:
        line:
            A line in the csv file.
        id_to_url_mapping:
            A dict mapping topic_id to
            the (url, title, description) of the topic.
        variance:
            The variance of the knowledge
            component of the topic.
        kc_init_func:
            The generator function of the knowledge component.

    Returns:
        A tuple (learner_id, event, label) where label indicates whether
        the learner engages in the event.
    """
    # unpack the data based on the format
    _, _, _, event_time, learner_id, *topics, label = line

    # sanitize the data
    event_time = float(event_time)
    learner_id = int(learner_id)
    label = bool(int(label))

    # extract topic_id from topics
    topic_ids = [int(float(topic_id)) for topic_id in topics[0::2]]
    # extract topic_skills from topics
    topic_skills = [float(topic_skill) for topic_skill in topics[1::2]]

    # remove -1 (empty topic_id and skill)
    topic_ids.append(-1)  # append -1 to avoid ValueError when calling .index
    topic_ids = topic_ids[: topic_ids.index(-1)]  # drop useless ids
    topic_skills = topic_skills[: len(topic_ids)]  # drop useless skills

    # construct the knowledge based on the topics and mapping
    knowledge = Knowledge(
        {
            topic_id: kc_init_func(
                mean=skill,
                variance=variance,
                timestamp=event_time,
                url=id_to_url_mapping[topic_id][0],
                title=id_to_url_mapping[topic_id][1],
                description=id_to_url_mapping[topic_id][2],
            )
            for topic_id, skill in zip(topic_ids, topic_skills)
        }
    )

    # add the constructed EventModel and label to the list of events
    # that belongs to the learner_id
    return learner_id, EventModel(knowledge=knowledge, event_time=event_time), label


def __restructure_data(
    filepath: str,
    id_to_url_mapping: Dict[int, Tuple[str, str, str]],
    variance: float,
    kc_init_func: PEEKKnowledgeComponentGenerator,
    limit: Optional[int],
) -> PEEKData:
    """Restructure data from the PEEKDataset.

    This function extracts the time, session (learner_id),
    topics, and label from the data, and constructs the appropriate
    model for this data.

    Args:
        filepath:
            The path of the PEEKDataset file.
            It should be a csv file.
        id_to_url_mapping:
            A dict mapping topic_id to
            the (url, title, description) of the topic.
        variance:
            The variance of the knowledge
            component of the topic.
        kc_init_func:
            The generator function of the knowledge component.
        limit:
            The number of lines to load from the file.
            If None, it means load all the data.

    Returns:
        PEEKData. PEEKData is a list of tuples (learner_id, events) where
        learner_id is the unique id that identifies a learner and events are
        a list of tuples (event, label) where event is an EventModel and
        label is a bool indicating whether the learner engages in this event.
    """
    with open(filepath, encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")

        learner_id_to_event_and_label: Dict[
            int, List[Tuple[EventModel, bool]]
        ] = collections.defaultdict(list)

        for line in itertools.islice(reader, limit):
            learner_id, event, label = __restructure_line(
                line, id_to_url_mapping, variance, kc_init_func
            )
            learner_id_to_event_and_label[learner_id].append((event, label))

        return list(learner_id_to_event_and_label.items())


def __build_mapping(mapping_filepath: str) -> Dict[int, Tuple[str, str, str]]:
    """Build the topic_id to url mapping.

    Args:
        mapping_filepath: The filepath of the id_to_url_mapping in the PEEKDataset.

    Returns:
        A dict mapping topic id to its corresponding (url, title, description).
    """
    with open(mapping_filepath, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader)
        return {
            int(topic_id): (str(url), str(title), str(description))
            for topic_id, url, title, description in csv_reader
        }


def __load_data_raw(filepath: str, limit: Optional[int]) -> List[List[str]]:
    """Load the PEEKDataset file without any processing.

    Args:
        filepath:
            The path of the PEEKDataset file.
            It should be a csv file.
        limit:
            The number of lines to load from the given file.
            If None, it means load all the data.

    Returns:
        A list of lines where line is represented as a
        list of string. A line is a row in the original file.
        The string is the value at the corresponding cell in
        the csv file.
    """
    with open(filepath, encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        return list(itertools.islice(reader, limit))


def load_peek_dataset(
    *,
    dirname: str = ".",
    variance: float = 1e-9,
    kc_init_func: PEEKKnowledgeComponentGenerator = KnowledgeComponent,
    train_limit: Optional[int] = None,
    test_limit: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[PEEKData, PEEKData, Dict[int, Tuple[str, str, str]]]:
    """Download and Parse PEEKDataset.

    Examples:
        To load the data:

        >>> from truelearn.datasets import load_peek_dataset
        >>> train, test, mapping = load_peek_dataset(verbose=False)
        >>> len(train)
        14050
        >>> train[0]  # doctest:+ELLIPSIS
        (23128, [(EventModel(...), event_time=172.0), False), ..., (EventModel(...), \
event_time=55932.0), False)])
        >>> len(test)
        5969
        >>> test[0]  # doctest:+ELLIPSIS
        (25623, [(EventModel(...), event_time=0.0), False), ..., (EventModel(...), \
event_time=1590.0), False)])
        >>> len(mapping)
        30367
        >>> mapping[0]
        ('https://en.wikipedia.org/wiki/"Hello,_World!"_program', \
'"Hello, World!" program', "Traditional beginners' computer program")

    Args:
        *:
            Use to reject positional arguments.
        dirname:
            The directory name.
        variance:
            The default variance of the knowledge components in PEEKDataset.
        kc_init_func:
            A function that creates a knowledge component.
            This can be customized to work with different kinds
            of knowledge components, as long as they follow the
            AbstractKnowledge protocol. The default is to initialize
            the KnowledgeComponent instance.
        train_limit:
            An optional non-negative integer specifying the
            maximum number of lines to read from the train file.
            If None, it means no limit.
        test_limit:
            An optional non-negative integer specifying the
            maximum number of lines to read from the test file.
            If None, it means no limit.
        verbose:
            If True and the downloaded file doesn't exist, this function outputs some
            information about the downloaded file.

    Returns:
        A tuple of (train, test, mapping) where train and test are PEEKData
        and mapping is a dict mapping topic_id to (url, title, description).
        PEEKData is a list of tuples (learner_id, events) where learner_id
        is the unique id that identifies a learner and events are a list of
        tuples (event, label) where event is an EventModel and label is a bool
        indicating whether the learner engages in this event.

        The returned data looks like this::

            (
                [
                    (leaner_id, [
                        (event, label), ...
                    ]),...
                ],
                [
                    ...
                ],
                {
                    0: (url, title, description),...  # 0 is wiki id
                }
            )

    Raises:
        TrueLearnValueError:
            If the train_limit or test_limit is less than 0.
    """
    __sanity_check(train_limit, test_limit)
    train_filepath, test_filepath, mapping_filepath = __download_files(dirname, verbose)

    # build mapping
    id_to_url_mapping = __build_mapping(mapping_filepath=mapping_filepath)

    # load peek dataset helper
    return (
        __restructure_data(
            filepath=train_filepath,
            id_to_url_mapping=id_to_url_mapping,
            variance=variance,
            kc_init_func=kc_init_func,
            limit=train_limit,
        ),
        __restructure_data(
            filepath=test_filepath,
            id_to_url_mapping=id_to_url_mapping,
            variance=variance,
            kc_init_func=kc_init_func,
            limit=test_limit,
        ),
        id_to_url_mapping,
    )


def load_peek_dataset_raw(
    *,
    dirname: str = ".",
    train_limit: Optional[int] = None,
    test_limit: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[List[List[str]], List[List[str]], Dict[int, Tuple[str, str, str]]]:
    """Download and Load the raw PEEKDataset.

    Examples:
        To load the data:

        >>> from truelearn.datasets import load_peek_dataset_raw
        >>> train, test, mapping = load_peek_dataset_raw(verbose=False)
        >>> len(train)
        203590
        >>> train[0]  # doctest:+ELLIPSIS
        ['4248', '1', '1', '172.0', ..., '1']
        >>> len(test)
        86945
        >>> test[0]  # doctest:+ELLIPSIS
        ['12730', '1', '1', '0.0', ..., '0']
        >>> len(mapping)
        30367
        >>> mapping[0]
        ('https://en.wikipedia.org/wiki/"Hello,_World!"_program', \
'"Hello, World!" program', "Traditional beginners' computer program")

    Args:
        *:
            Use to reject positional arguments.
        dirname:
            The directory name.
        train_limit:
            An optional non-negative integer specifying the
            maximum number of lines to read from the train file.
            If None, it means no limit.
        test_limit:
            An optional non-negative integer specifying the
            maximum number of lines to read from the test file.
            If None, it means no limit.
        verbose:
            If True and the downloaded file doesn't exist, this function outputs some
            information about the downloaded file.

    Returns:
        A tuple of (train, test, mapping) where train and test are list of
        lines and mapping is a dict mapping topic_id to (url, title, description).
        Each line in train and test is a list of strings where each string is
        the value of the cell in the original csv.

        The data looks like this::

            (
                [
                    [slug, vid_id, ...],...  # follow the structure of header
                ],
                [
                    ...
                ],
                {
                    0: (url, title, description),...  # 0 is wiki id
                }
            )

    Raises:
        TrueLearnValueError:
            If the train_limit or test_limit is less than 0.
    """
    __sanity_check(train_limit, test_limit)
    train_filepath, test_filepath, mapping_filepath = __download_files(dirname, verbose)

    return (
        __load_data_raw(filepath=train_filepath, limit=train_limit),
        __load_data_raw(
            filepath=test_filepath,
            limit=test_limit,
        ),
        __build_mapping(mapping_filepath=mapping_filepath),
    )
