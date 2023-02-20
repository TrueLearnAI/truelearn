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
from typing import Tuple, Dict, List, Optional
from typing_extensions import TypeAlias, Protocol

from truelearn.models import (
    EventModel,
    Knowledge,
    KnowledgeComponent,
    AbstractKnowledgeComponent,
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
    "/1740aa04aeb019494fd3593d4f708fd519fa101a/datasets/v1/id_to_wiki_url_mapping.csv",
    filename="peek.mapping.csv",
    expected_sha256="568e23b43517dea6649cc140e492468717c53e7f7d97e42f9a4672b9becba835",
)


class PEEKKnowledgeComponentGenerator(Protocol):
    """A class that defines an implicit interface\
    to the generator of the knowledge component."""

    def __call__(
        self, mean: float, variance: float, timestamp: float, url: str
    ) -> AbstractKnowledgeComponent:
        ...


# pylint: disable=too-many-locals
def _restructure_data(
    filepath: str,
    id_to_url_mapping: Dict[int, str],
    variance: float,
    kc_init_func: PEEKKnowledgeComponentGenerator,
    limit: int,
) -> PEEKData:
    """Restructure data from the PEEKDataset.

    This function extracts the time, session (learner_id),
    topics, and label from the data, and constructs the appropriate
    the model for these data.

    Args:
        filepath:
            The path of the PEEKDataset file.
            It should be a csv file.
        id_to_url_mapping:
            A dict mapping topic_id to
            the url of the topic.
        variance:
            The variance of the knowledge
            component of the topic.
        kc_init_func:
            The generator function of the knowledge component.
        limit:
            The number of lines to load from the file.

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

        for line in reader:
            if limit == 0:
                break

            # unpack the data based on the format
            _, _, _, event_time, learner_id, *topics, label = line

            # sanitize the data
            event_time = float(event_time)
            learner_id = int(learner_id)
            label = bool(label)

            # extract topic_id from topics
            topic_ids = list(
                map(
                    lambda topic_id: int(float(topic_id[1])),
                    filter(lambda idx: idx[0] % 2 == 0, enumerate(topics)),
                )
            )
            # extract topic_skills from topics
            topic_skills = list(
                map(
                    lambda topic_skill: float(topic_skill[1]),
                    filter(lambda idx: idx[0] % 2 == 1, enumerate(topics)),
                )
            )

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
                        url=id_to_url_mapping[topic_id],
                    )
                    for topic_id, skill in zip(topic_ids, topic_skills)
                }
            )

            # add the constructed EventModel and label to the list of events
            # that belongs to the learner_id
            learner_id_to_event_and_label[learner_id].append(
                (EventModel(knowledge=knowledge, event_time=event_time), label)
            )

            limit -= 1

        return list(learner_id_to_event_and_label.items())


def _build_mapping(mapping_filepath: str) -> Dict[int, str]:
    """Build the topic_id to url mapping.

    Args:
        mapping_filepath: The filepath of the id_to_url_mapping in the PEEKDataset.

    Returns:
        A dict mapping id to its corresponding url.
    """
    with open(mapping_filepath, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader)
        return {int(topic_id): str(url) for topic_id, url in csv_reader}


def _get_knowledge_component(
    mean: float, variance: float, timestamp: float, url: str
) -> KnowledgeComponent:
    """The default generator function of the knowledge component.

    Args:
        mean: The mean of the generated KnowledgeComponent.
        variance: The variance of the generated KnowledgeComponent.
        timestamp: The timestamp of the generated KnowledgeComponent.
        url: The url of the generated KnowledgeComponent.

    Returns:
        The generated KnowledgeComponent.
    """
    return KnowledgeComponent(
        mean=mean, variance=variance, timestamp=timestamp, url=url
    )


def load_peek_dataset(
    *,
    dirname: Optional[str] = ".",
    variance: float = 1e-9,
    kc_init_func: PEEKKnowledgeComponentGenerator = _get_knowledge_component,
    train_limit: int = -1,
    test_limit: int = -1,
) -> Tuple[PEEKData, PEEKData, Dict[int, str]]:
    """Download and Parse PEEKDataset.

    Args:
        *:
            Use to reject positional arguments.
        dirname:
            The directory name. Defaults to ".".
        variance:
            The default variance of the knowledge components in PEEKDataset.
        kc_init_func:
            A function that creates a knowledge component.
            This can be customized to work with different kinds
            of knowledge components, as long as they follow the
            AbstractKnowledge protocol. The default is to initialize
            the KnowledgeComponent instance.
        train_limit:
            The number of lines to load from the training data.
            A negative number denotes unlimited (load all).
            Defaults to -1.
        test_limit:
            The number of lines to load from the testing data.
            A negative number denotes unlimited (load all).
            Defaults to -1.

    Returns:
        A tuple of (train, test, mapping) where train and test are PEEKData
        and mapping is a dict mapping topic_id to url. PEEKData is a list of
        tuples (learner_id, events) where learner_id is the unique id that
        identifies a learner and events are a list of tuples (event, label)
        where event is an EventModel and label is a bool indicating whether
        the learner engages in this event.

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
                    0: "url",...  # 0 is wiki id
                }
            )
    """
    train_filepath = check_and_download_file(remote_file=PEEK_TRAIN, dirname=dirname)
    test_filepath = check_and_download_file(remote_file=PEEK_TEST, dirname=dirname)
    mapping_filepath = check_and_download_file(
        remote_file=PEEK_MAPPING, dirname=dirname
    )

    # build mapping
    id_to_url_mapping = _build_mapping(mapping_filepath=mapping_filepath)

    # load peek dataset helper
    return (
        _restructure_data(
            filepath=train_filepath,
            id_to_url_mapping=id_to_url_mapping,
            variance=variance,
            kc_init_func=kc_init_func,
            limit=train_limit,
        ),
        _restructure_data(
            filepath=test_filepath,
            id_to_url_mapping=id_to_url_mapping,
            variance=variance,
            kc_init_func=kc_init_func,
            limit=test_limit,
        ),
        id_to_url_mapping,
    )


def _load_data_raw(filepath: str, limit: int) -> List[List[str]]:
    """Load the PEEKDataset file without any processing.

    Args:
        filepath: The path of the PEEKDataset file.
        It should be a csv file.

    Returns:
        A list of lines where line is represented as a
        list of string. A line is a row in the original file.
        The string is the value at the corresponding cell in
        the csv file.
    """
    with open(filepath, encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        lines = []
        for line in reader:
            if limit == 0:
                break
            lines.append(line)
            limit -= 1
        return lines


def load_peek_dataset_raw(
    *,
    dirname: Optional[str] = ".",
    train_limit: int = -1,
    test_limit: int = -1,
) -> Tuple[List[List[str]], List[List[str]], Dict[int, str]]:
    """Download and Load the raw PEEKDataset.

    Args:
        *:
            Use to reject positional arguments.
        dirname:
            The directory name. Defaults to ".".
        train_limit:
            The number of lines to load from the training data.
            A negative number denotes unlimited (load all).
            Defaults to -1.
        test_limit:
            The number of lines to load from the testing data.
            A negative number denotes unlimited (load all).
            Defaults to -1.

    Returns:
        A tuple of (train, test, mapping) where train and test are list of
        lines and mapping is a dict mapping topic_id to url.
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
                    0: "url",...  # 0 is wiki id
                }
            )
    """
    train_filepath = check_and_download_file(remote_file=PEEK_TRAIN, dirname=dirname)
    test_filepath = check_and_download_file(remote_file=PEEK_TEST, dirname=dirname)
    mapping_filepath = check_and_download_file(
        remote_file=PEEK_MAPPING, dirname=dirname
    )

    return (
        _load_data_raw(filepath=train_filepath, limit=train_limit),
        _load_data_raw(
            filepath=test_filepath,
            limit=test_limit,
        ),
        _build_mapping(mapping_filepath=mapping_filepath),
    )
