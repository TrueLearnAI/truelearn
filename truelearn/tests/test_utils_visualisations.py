# pylint: disable=missing-function-docstring,missing-class-docstring
import functools
import random
import pathlib
import filecmp
import types
import os
import sys
from typing import Dict, Optional, Callable

import pytest
from matplotlib.testing.compare import compare_images

from truelearn import learning, datasets, models
from truelearn.utils import visualisations
from truelearn.errors import TrueLearnTypeError


BASELINE_DIR = (
    pathlib.Path(__file__).parent / "baseline_files"
)  # directory relative to truelearn/tests
TMP_PATH = pathlib.Path("./tests")  # directory relative to root
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

    # a number that breaks this knowledge into a small and a large part
    # so that we can provide 3 knowledge of different sizes
    bound = 3

    knowledge_small = models.Knowledge(dict(list(knowledge.topic_kc_pairs())[:bound]))
    knowledge_large = models.Knowledge(dict(list(knowledge.topic_kc_pairs())[bound:]))

    yield knowledge, knowledge_large, knowledge_small

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


def _file_comparison_method_wrapper_generator(
    func,
    extensions: Dict[str, Dict],
    file_cmp_func: Callable[[str, str], bool],
    tmp_path_dir: pathlib.Path,
    target_path_dir: pathlib.Path,
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

            if not file_cmp_func(str(target_file), str(tmp_file)):
                failed_ext_with_reasons[ext] = "Tmp file does not match target file."
                continue

            # remove images that pass the test
            os.remove(tmp_file)

        if failed_ext_with_reasons:
            raise ValueError(
                "The file generated with the following extension does not "
                f"match the baseline file {failed_ext_with_reasons!r}"
            )

    return wrapper


def file_comparison(plotter_type: str, config: Optional[Dict[str, Dict]] = None):
    """Class decorator for image comparison.

    Args:
        plotter_type:
            The plotter type. Supported plotter types are: "plotly", "matplotlib".
            Based on the given type, the method will determine the file format
            to be tested and how to compare the resulting files.
            For plotly type, the method will test `.html` and `.json`.
            For matplotlib type, the method will test `.png`.
        config:
            A dictionary containing the configuration for each extension.
    """
    config = config or {}

    if plotter_type == "plotly":
        # only support html and json for plotly
        # because the backend engine that plotly uses
        # to generate imgaes is platform dependent
        # Therefore, to be able to provide consistent
        # and replicable tests, we test against json and html.
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
            # for images, we only require them to be similar within a tolerance
            return compare_images(filename1, filename2, tol=0.1) is None

    def file_comparison_class_decorator(tclass):
        target_path = BASELINE_DIR / str(tclass.__name__).lower()
        tmp_path = TMP_PATH / str(tclass.__name__).lower()
        tmp_path.mkdir(parents=True, exist_ok=True)

        for key in dir(tclass):
            value = getattr(tclass, key)
            if isinstance(value, types.FunctionType):
                wrapped = _file_comparison_method_wrapper_generator(
                    value, extensions, file_cmp_func, tmp_path, target_path
                )
                setattr(tclass, key, wrapped)

        return tclass

    return file_comparison_class_decorator


@file_comparison(plotter_type="plotly")
class TestBasePlotter:
    def test_standardise_data_with_topics_via_bar(self, resources):
        plotter = visualisations.BarPlotter()
        plotter.plot(resources[0], topics=["Posterior Probability", "Algorithm"])
        return plotter


class TestBasePlotterNormal:
    def test_no_history_throw_via_bar_history(self):
        plotter = visualisations.BarPlotter()

        with pytest.raises(TrueLearnTypeError) as excinfo:
            plotter.plot(
                models.Knowledge(
                    {1: models.KnowledgeComponent(mean=0.0, variance=0.5)}
                ),
                history=True,
            )
        assert (
            str(excinfo.value) == "Learner's knowledge does not contain history. "
            "You can use HistoryAwareKnowledgeComponents."
        )


class TestPlotlyBasePlotter:
    def test_show_no_error_via_bar(self):
        plotter = visualisations.BarPlotter()
        plotter.show()

    def test_savefig_no_error(self, resources, tmp_path):
        plotter = visualisations.BarPlotter()
        plotter.plot(resources[0])
        plotter.savefig(
            os.path.join(tmp_path, "TestPlotlyBasePlotter.test_savefig_no_error.png")
        )
        plotter.savefig(
            os.path.join(tmp_path, "TestPlotlyBasePlotter.test_savefig_no_error.jpg")
        )
        plotter.savefig(
            os.path.join(tmp_path, "TestPlotlyBasePlotter.test_savefig_no_error.jpeg")
        )
        plotter.savefig(
            os.path.join(tmp_path, "TestPlotlyBasePlotter.test_savefig_no_error.svg")
        )
        plotter.savefig(
            os.path.join(tmp_path, "TestPlotlyBasePlotter.test_savefig_no_error.pdf")
        )
        plotter.savefig(
            os.path.join(tmp_path, "TestPlotlyBasePlotter.test_savefig_no_error.html")
        )
        plotter.savefig(
            os.path.join(tmp_path, "TestPlotlyBasePlotter.test_savefig_no_error.json")
        )


class TestMatplotlibBasePlotter:
    def test_show_no_error_via_bubble(self):
        plotter = visualisations.BubblePlotter()
        plotter.show()

    def test_savefig_no_error(self, resources, tmp_path):
        plotter = visualisations.BubblePlotter()
        plotter.plot(resources[2])
        plotter.savefig(
            os.path.join(
                tmp_path, "TestMatplotlibBasePlotter.test_savefig_no_error.png"
            )
        )
        plotter.savefig(
            os.path.join(
                tmp_path, "TestMatplotlibBasePlotter.test_savefig_no_error.jpg"
            )
        )
        plotter.savefig(
            os.path.join(
                tmp_path, "TestMatplotlibBasePlotter.test_savefig_no_error.jpeg"
            )
        )
        plotter.savefig(
            os.path.join(
                tmp_path, "TestMatplotlibBasePlotter.test_savefig_no_error.svg"
            )
        )
        plotter.savefig(
            os.path.join(
                tmp_path, "TestMatplotlibBasePlotter.test_savefig_no_error.pdf"
            )
        )


@file_comparison(plotter_type="plotly")
class TestBarPlotter:
    def test_default(self, resources):
        plotter = visualisations.BarPlotter()
        plotter.plot(resources[0])
        return plotter

    def test_history(self, resources):
        plotter = visualisations.BarPlotter()
        plotter.plot(resources[0], history=True)
        return plotter

    def test_empty_knowledge(self):
        plotter = visualisations.BarPlotter()
        plotter.plot(models.Knowledge())
        return plotter

    def test_top_n(self, resources):
        plotter = visualisations.BarPlotter()
        plotter.plot(resources[0], top_n=3)
        return plotter


@file_comparison(
    plotter_type="matplotlib",
)
class TestBubblePlotter:
    def test_default(self, resources):
        plotter = visualisations.BubblePlotter()
        plotter.plot(resources[2])
        return plotter

    def test_empty_knowledge(self):
        plotter = visualisations.BubblePlotter()
        plotter.plot(models.Knowledge())
        return plotter

    def test_top_n(self, resources):
        plotter = visualisations.BubblePlotter()
        plotter.plot(resources[0], top_n=3)
        return plotter


@file_comparison(plotter_type="plotly")
class TestDotPlotter:
    def test_default(self, resources):
        plotter = visualisations.DotPlotter()
        plotter.plot(resources[2])
        return plotter

    def test_history(self, resources):
        plotter = visualisations.DotPlotter()
        plotter.plot(resources[2], history=True)
        return plotter

    def test_empty_knowledge(self):
        plotter = visualisations.DotPlotter()
        plotter.plot(models.Knowledge())
        return plotter

    def test_top_n(self, resources):
        plotter = visualisations.DotPlotter()
        plotter.plot(resources[0], top_n=3)
        return plotter


@file_comparison(plotter_type="plotly")
class TestLinePlotterSingleUser:
    def test_default(self, resources):
        plotter = visualisations.LinePlotter()
        plotter.plot(resources[0])
        return plotter

    def test_empty_knowledge(self):
        plotter = visualisations.LinePlotter()
        plotter.plot(models.Knowledge())
        return plotter

    def test_top_n(self, resources):
        plotter = visualisations.LinePlotter()
        plotter.plot(resources[0], top_n=3)
        return plotter


class TestLinePlotterSingleUserNormal:
    def test_no_history_throw(self):
        plotter = visualisations.LinePlotter()

        with pytest.raises(TrueLearnTypeError) as excinfo:
            plotter.plot(
                models.Knowledge({1: models.KnowledgeComponent(mean=0.0, variance=0.5)})
            )
        assert (
            str(excinfo.value) == "Learner's knowledge does not contain history. "
            "You can use HistoryAwareKnowledgeComponents."
        )

    def test_iterator_topics_no_throw(self):
        plotter = visualisations.LinePlotter()
        topics = ["Machine learning", "Probability"]
        topics = map(lambda x: x, topics)

        plotter.plot(
            models.Knowledge(
                {1: models.HistoryAwareKnowledgeComponent(mean=0.0, variance=0.5)}
            ),
            topics=topics,
        )


@file_comparison(plotter_type="plotly")
class TestLinePlotterMultipleUsers:
    def test_default(self, resources):
        plotter = visualisations.LinePlotter()
        plotter.plot([resources[1], resources[2]])
        return plotter

    def test_empty_knowledge(self, resources):
        plotter = visualisations.LinePlotter()
        plotter.plot([resources[2], models.Knowledge()])
        return plotter

    def test_empty_list_of_knowledge(self):
        plotter = visualisations.LinePlotter()
        plotter.plot([])
        return plotter


@file_comparison(plotter_type="plotly")
class TestPiePlotter:
    def test_default(self, resources):
        plotter = visualisations.PiePlotter()
        plotter.plot(resources[2])
        return plotter

    def test_history(self, resources):
        plotter = visualisations.PiePlotter()
        plotter.plot(resources[2], history=True)
        return plotter

    def test_other(self, resources):
        plotter = visualisations.PiePlotter()
        plotter.plot(resources[2], other=True)
        return plotter

    def test_history_and_other(self, resources):
        plotter = visualisations.PiePlotter()
        plotter.plot(resources[2], history=True, other=True)
        return plotter

    def test_empty_knowledge(self):
        plotter = visualisations.PiePlotter()
        plotter.plot(models.Knowledge())
        return plotter

    def test_top_n(self, resources):
        plotter = visualisations.PiePlotter()
        plotter.plot(resources[0], top_n=3)
        return plotter


@file_comparison(plotter_type="plotly")
class TestRosePlotter:
    def test_default(self, resources):
        random_state = random.Random(42)
        plotter = visualisations.RosePlotter()
        plotter.plot(resources[2], random_state=random_state)
        return plotter

    def test_other(self, resources):
        random_state = random.Random(42)
        plotter = visualisations.RosePlotter()
        plotter.plot(resources[2], other=True, random_state=random_state)
        return plotter

    def test_empty_knowledge(self):
        random_state = random.Random(42)
        plotter = visualisations.RosePlotter()
        plotter.plot(models.Knowledge(), random_state=random_state)
        return plotter

    def test_top_n(self, resources):
        random_state = random.Random(42)
        plotter = visualisations.RosePlotter()
        plotter.plot(resources[0], top_n=3, random_state=random_state)
        return plotter


@file_comparison(plotter_type="plotly")
class TestRadarPlotter:
    def test_default(self, resources):
        plotter = visualisations.RadarPlotter()
        plotter.plot(resources[2])
        return plotter

    def test_empty_knowledge(self):
        plotter = visualisations.RadarPlotter()
        plotter.plot(models.Knowledge())
        return plotter

    def test_top_n(self, resources):
        plotter = visualisations.RadarPlotter()
        plotter.plot(resources[0], top_n=3)
        return plotter


@file_comparison(plotter_type="plotly")
class TestTreePlotter:
    def test_default(self, resources):
        plotter = visualisations.TreePlotter()
        plotter.plot(resources[0])
        return plotter

    def test_history(self, resources):
        plotter = visualisations.TreePlotter()
        plotter.plot(resources[0], history=True)
        return plotter

    def test_empty_knowledge(self):
        plotter = visualisations.TreePlotter()
        plotter.plot(models.Knowledge())
        return plotter

    def test_top_n(self, resources):
        plotter = visualisations.TreePlotter()
        plotter.plot(resources[0], top_n=3)
        return plotter


@pytest.mark.skipif(
    sys.version_info >= (3, 8),
    reason="WordPlotter only supports Python version <= 3.7",
)
class TestWordPlotter:
    # no way to test this in a cross-platform way
    # because wordcloud renders the same font differently on different operating systems
    def test_generate_files_no_error(self, resources):
        random_state = random.Random(42)
        plotter = visualisations.WordPlotter()
        plotter.plot(resources[0], random_state=random_state)

    def test_empty_knowledge_no_throw(self):
        random_state = random.Random(42)
        plotter = visualisations.WordPlotter()
        plotter.plot(models.Knowledge(), random_state=random_state)
