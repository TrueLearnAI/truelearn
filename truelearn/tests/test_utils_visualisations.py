# pylint: disable=missing-function-docstring,missing-class-docstring
import functools
import random
import pathlib
import types
import os
import sys

import pytest
from matplotlib.testing.compare import compare_images

from truelearn import learning, datasets, models
from truelearn.utils import visualisations


BASELINE_DIR = "baseline_files"  # directory relative to truelearn/tests
TMP_PATH = pathlib.Path("./tests")  # directory relative to .


@pytest.fixture(scope="module")
def resources():
    # prepare data
    # store only one copy
    train, _, _ = datasets.load_peek_dataset(
        test_limit=0, kc_init_func=models.HistoryAwareKnowledgeComponent
    )
    learning_events = train[12][1]

    classifier = learning.KnowledgeClassifier()
    for event, label in learning_events:
        classifier.fit(event, label)

    knowledge = classifier.get_learner_model().knowledge

    # separate that knowledge into two parts, so we can test plotter
    # that utilizes multiple learners
    BOUND = 3
    knowledge_small = models.Knowledge(dict(list(knowledge.topic_kc_pairs())[:BOUND]))
    knowledge_large = models.Knowledge(dict(list(knowledge.topic_kc_pairs())[BOUND:]))

    yield (knowledge, knowledge_large, knowledge_small)

    try:
        dirs_to_remove = os.listdir(TMP_PATH)
    except OSError:
        ...
    else:
        dirs_to_remove = [TMP_PATH / directory for directory in dirs_to_remove]
        dirs_to_remove.append(TMP_PATH)

        # cleanup TMP_DIR (remove as many directories as possible)
        for directory in dirs_to_remove:
            try:
                directory.rmdir()
            except (FileNotFoundError, OSError):  # contain files
                ...


def image_comparison():
    """Class decorator for image comparison.

    Args:
        func:
            The function to decorate.
        type_of_plotter:
            The plotter type. Supported plotter types are: "plotly", "matplotlib"
    """
    # can support more file types if:
    # - use custom way to convert each type to png
    # - use compare_images to check if they are similar within some tol
    extensions = [".png"]

    def image_comparison_class_decorator(tclass):
        # only works for class decorator
        assert isinstance(tclass, type)

        def image_compression_method_decorator(
            func, tmp_path_dir: pathlib.Path, target_path_dir: pathlib.Path
        ):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                plotter = func(*args, **kwargs)
                failed_ext_with_reasons = {}

                for ext in extensions:
                    tmp_file = tmp_path_dir / str(func.__name__ + ext)
                    target_file = target_path_dir / str(func.__name__ + ext)

                    plotter.savefig(str(tmp_file))

                    if not target_file.exists():
                        failed_ext_with_reasons[ext] = "Target file does not exist."
                        continue

                    if compare_images(str(target_file), str(tmp_file), 0.1):
                        failed_ext_with_reasons[
                            ext
                        ] = "Tmp file does not match target file."
                        continue

                    # remove images that pass the test
                    os.remove(tmp_file)

                if failed_ext_with_reasons:
                    raise ValueError(
                        "The file generated with the following extension does not "
                        f"match the baseline file {failed_ext_with_reasons!r}"
                    )

            return wrapper

        target_path = (
            pathlib.Path(__file__).parent / BASELINE_DIR / str(tclass.__name__).lower()
        )
        tmp_path = TMP_PATH / str(tclass.__name__).lower()
        tmp_path.mkdir(parents=True, exist_ok=True)

        for key in dir(tclass):
            value = getattr(tclass, key)
            if isinstance(value, types.FunctionType):
                wrapped = image_compression_method_decorator(
                    value, tmp_path, target_path
                )
                setattr(tclass, key, wrapped)

        # method decorator
        return tclass

    return image_comparison_class_decorator


@image_comparison()
class TestBarPlotter:
    def test_default(self, resources):
        plotter = visualisations.BarPlotter()
        plotter.plot(resources[0])
        return plotter


@image_comparison()
class TestBubblePlot:
    def test_default(self, resources):
        plotter = visualisations.BubblePlotter()
        plotter.plot(resources[2])
        return plotter


@image_comparison()
class TestDotPlotter:
    def test_default(self, resources):
        plotter = visualisations.DotPlotter()
        plotter.plot(resources[2])
        return plotter


@image_comparison()
class TestLinePlotterSingleUser:
    def test_default(self, resources):
        plotter = visualisations.LinePlotter()
        plotter.plot(resources[0])
        return plotter


@image_comparison()
class TestLinePlotterMultipleUsers:
    def test_default(self, resources):
        plotter = visualisations.LinePlotter()
        plotter.plot([resources[1], resources[2]])
        return plotter


@image_comparison()
class TestPiePlotter:
    def test_default(self, resources):
        plotter = visualisations.PiePlotter()
        plotter.plot(resources[2])
        return plotter


@image_comparison()
class TestRosePlotter:
    def test_default(self, resources):
        random_state = random.Random(42)
        plotter = visualisations.RosePlotter()
        plotter.plot(resources[2], random_state=random_state)
        return plotter


@image_comparison()
class TestRadarPlotter:
    def test_default(self, resources):
        plotter = visualisations.RadarPlotter()
        plotter.plot(resources[2])
        return plotter


@image_comparison()
class TestTreePlotter:
    def test_default(self, resources):
        plotter = visualisations.TreePlotter()
        plotter.plot(resources[0])
        return plotter


@image_comparison()
@pytest.mark.skipif(
    sys.version_info >= (3, 10),
    reason="WordPlotter only supports Python version < 3.10",
)
class TestWordPlotter:
    def test_default(self, resources):
        random_state = random.Random(42)
        plotter = visualisations.WordPlotter()
        plotter.plot(resources[0], random_state=random_state)
        return plotter
