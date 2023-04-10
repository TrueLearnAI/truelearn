# pylint: disable=missing-function-docstring,missing-class-docstring
import functools
import random
import pathlib
import filecmp
import types
import os
import sys
from typing import Dict, Optional

import pytest
from matplotlib.testing.compare import compare_images

from truelearn import learning, datasets, models
from truelearn.utils import visualisations


BASELINE_DIR = (
    pathlib.Path(__file__).parent / "baseline_files"
)  # directory relative to truelearn/tests
TMP_PATH = pathlib.Path("./tests")  # directory relative to .
UUID = "71903f8e-0ae2-4fd1-9c0c-2290e95b21e9"  # a randomly chosen UUID, affect HTML gen


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


def file_comparison(plotter_type: str, config: Optional[Dict[str, Dict]] = None):
    """Class decorator for image comparison.

    Args:
        plotter_type:
            The plotter type. Supported plotter types are: "plotly", "matplotlib"
        config:
            A dictionary containing the configuration for each extension.
    """
    config = config or {}

    if plotter_type == "plotly":
        # only support html and json for plotly
        # because the backend engine that plotly uses
        # to generate imgaes is platform dependent.
        # (Because it uses Chrome).
        # Therefore, there is no way to generate consistent
        # images cross different platforms.
        extensions = {
            ".json": config.get(".json", {}),
            ".html": {
                **config.get(".html", {}),
                # overwrite settings for div_id and include_plotlyjs
                # as they directly affect the generated output
                "div_id": UUID,
                "include_plotlyjs": "https://cdn.plot.ly/plotly-2.20.0.min.js",
            },
        }

        def file_cmp_func(filename1, filename2):
            return filecmp.cmp(filename1, filename2)

    elif plotter_type == "matplotlib":
        extensions = {
            ".png": config.get(".png", {}),
        }

        def file_cmp_func(filename1, filename2):
            return compare_images(filename1, filename2, tol=0.1) is None

    def file_compression_method_decorator(
        func, tmp_path_dir: pathlib.Path, target_path_dir: pathlib.Path
    ):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            plotter = func(*args, **kwargs)
            failed_ext_with_reasons = {}

            for ext, config in extensions.items():
                tmp_file = tmp_path_dir / str(func.__name__ + ext)
                target_file = target_path_dir / str(func.__name__ + ext)

                plotter.savefig(str(tmp_file), **config)

                if not target_file.exists():
                    failed_ext_with_reasons[ext] = "Target file does not exist."
                    continue

                # if compare_images(str(target_file), str(tmp_file), tol=0):
                #     failed_ext_with_reasons[
                #         ext
                #     ] = "Tmp file does not match target file."
                #     continue

                if not file_cmp_func(str(target_file), str(tmp_file)):
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

    def file_comparison_class_decorator(tclass):
        target_path = BASELINE_DIR / str(tclass.__name__).lower()
        tmp_path = TMP_PATH / str(tclass.__name__).lower()
        tmp_path.mkdir(parents=True, exist_ok=True)

        for key in dir(tclass):
            value = getattr(tclass, key)
            if isinstance(value, types.FunctionType):
                wrapped = file_compression_method_decorator(
                    value, tmp_path, target_path
                )
                setattr(tclass, key, wrapped)

        return tclass

    return file_comparison_class_decorator


@file_comparison(plotter_type="plotly")
class TestBarPlotter:
    def test_default(self, resources):
        plotter = visualisations.BarPlotter()
        plotter.plot(resources[0])
        plotter.figure.update_layout()
        return plotter


@file_comparison(
    plotter_type="matplotlib",
)
class TestBubblePlotter:
    def test_default(self, resources):
        plotter = visualisations.BubblePlotter()
        plotter.plot(resources[2])
        return plotter


@file_comparison(plotter_type="plotly")
class TestDotPlotter:
    def test_default(self, resources):
        plotter = visualisations.DotPlotter()
        plotter.plot(resources[2])
        return plotter


@file_comparison(plotter_type="plotly")
class TestLinePlotterSingleUser:
    def test_default(self, resources):
        plotter = visualisations.LinePlotter()
        plotter.plot(resources[0])
        return plotter


@file_comparison(plotter_type="plotly")
class TestLinePlotterMultipleUsers:
    def test_default(self, resources):
        plotter = visualisations.LinePlotter()
        plotter.plot([resources[1], resources[2]])
        return plotter


@file_comparison(plotter_type="plotly")
class TestPiePlotter:
    def test_default(self, resources):
        plotter = visualisations.PiePlotter()
        plotter.plot(resources[2])
        return plotter


@file_comparison(plotter_type="plotly")
class TestRosePlotter:
    def test_default(self, resources):
        random_state = random.Random(42)
        plotter = visualisations.RosePlotter()
        plotter.plot(resources[2], random_state=random_state)
        return plotter


@file_comparison(plotter_type="plotly")
class TestRadarPlotter:
    def test_default(self, resources):
        plotter = visualisations.RadarPlotter()
        plotter.plot(resources[2])
        return plotter


@file_comparison(plotter_type="plotly")
class TestTreePlotter:
    def test_default(self, resources):
        plotter = visualisations.TreePlotter()
        plotter.plot(resources[0])
        return plotter


@pytest.mark.skipif(
    sys.version_info >= (3, 10),
    reason="WordPlotter only supports Python version < 3.10",
)
@file_comparison(
    plotter_type="matplotlib",
)
class TestWordPlotter:
    def test_default(self, resources):
        random_state = random.Random(42)
        plotter = visualisations.WordPlotter()
        plotter.plot(resources[0], random_state=random_state)
        return plotter
