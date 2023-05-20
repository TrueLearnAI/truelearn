# pylint: disable=missing-function-docstring,missing-class-docstring,protected-access
import math

import pytest

from truelearn import base, learning, models
from truelearn import _constraint
from truelearn.errors import (
    InvalidArgumentError,
    TrueLearnTypeError,
    TrueLearnValueError,
)


def check_farray_close(farr1, farr2):
    status = (math.isclose(n, m) for n, m in zip(farr1, farr2))

    if not all(status):
        assert False, "farr1 is not equal to farr2.\n" f"{farr1} != {farr2}"


class MockClassifier(base.BaseClassifier):
    HOLDER_VALUE = 42
    _parameter_constraints = {
        **base.BaseClassifier._parameter_constraints,
        "holder": _constraint.TypeConstraint(int),
    }

    def __init__(self):  # noqa: D107
        self._holder = self.HOLDER_VALUE

    def fit(self, _x, _y):
        ...

    def predict(self, _x):
        ...

    def predict_proba(self, _x):
        ...


class TestBase:
    def test_useless_key_with_get_params_skip(self, monkeypatch):
        monkeypatch.setattr(
            MockClassifier,
            "_parameter_constraints",
            {**MockClassifier._parameter_constraints, "useless_key": type(None)},
        )

        classifier = MockClassifier()
        classifier.get_params()

    def test_set_params_empty_no_effect(self):
        classifier = MockClassifier()
        params = classifier.get_params()
        classifier.set_params()

        assert (
            params == classifier.get_params() == {"holder": MockClassifier.HOLDER_VALUE}
        )

    def test_useless_key_with_validate_params_skip(self, monkeypatch):
        monkeypatch.setattr(
            MockClassifier,
            "_parameter_constraints",
            {
                **MockClassifier._parameter_constraints,
                "useless_key": _constraint.TypeConstraint(type(None)),
            },
        )

        classifier = MockClassifier()
        classifier._validate_params()

    def test_validate_params_type_mismatch_throw(self, monkeypatch):
        parameter_constraints = {
            **MockClassifier._parameter_constraints,
            "holder": _constraint.TypeConstraint(float, str, type(None)),
        }

        monkeypatch.setattr(
            MockClassifier,
            "_parameter_constraints",
            parameter_constraints,
        )

        classifier = MockClassifier()
        with pytest.raises(TrueLearnTypeError) as excinfo:
            classifier._validate_params()
        assert (
            str(excinfo.value) == "The holder parameter of "
            "class 'MockClassifier' "
            "must be one of the classes in ['float', 'str', 'NoneType']. "
            "Got 'int' instead."
        )


class TestEngageClassifier:
    def test_engage_classifier(self):
        classifier = learning.EngageClassifier()

        # no exception should occur
        classifier.fit(models.EventModel(), True)

        assert classifier.predict(models.EventModel())
        assert not classifier.get_params()
        assert classifier.predict_proba(models.EventModel()) == 1.0


class TestMajorityClassifier:
    def test_majority_classifier_fit_predict(self):
        classifier = learning.MajorityClassifier()

        # no exception should occur
        classifier.fit(models.EventModel(), True)

        assert classifier.predict(models.EventModel())

        assert classifier.predict_proba(models.EventModel()) == 1.0

        classifier.fit(models.EventModel(), False)

        # because engagement == non_engagement
        assert not classifier.predict(models.EventModel())
        assert classifier.predict_proba(models.EventModel()) == 0.5

    def test_majority_get_set_params(self):
        classifier = learning.MajorityClassifier()

        assert classifier.get_params() == {
            "engagement": 0,
            "non_engagement": 0,
            "threshold": 0.5,
        }

        classifier.set_params(engagement=1)
        assert classifier.get_params() == {
            "engagement": 1,
            "non_engagement": 0,
            "threshold": 0.5,
        }

        with pytest.raises(TrueLearnTypeError) as excinfo:
            classifier.set_params(non_engagement=1.0)
        assert (
            str(excinfo.value) == "The non_engagement parameter of "
            "class 'MajorityClassifier' "
            "must be one of the classes in ['int']. "
            "Got 'float' instead."
        )


class TestPersistentClassifier:
    def test_persistent_classifier_fit_predict(self):
        classifier = learning.PersistentClassifier()

        # no exception should occur
        classifier.fit(models.EventModel(), False)

        assert not classifier.predict(models.EventModel())
        assert classifier.get_params() == {
            "engage_with_last": False,
        }
        assert classifier.predict_proba(models.EventModel()) == 0.0

        classifier.fit(models.EventModel(), True)

        # because engage_with_last == True
        assert classifier.predict(models.EventModel())
        assert classifier.predict_proba(models.EventModel()) == 1.0

    def test_persistent_get_set_params(self):
        classifier = learning.PersistentClassifier()

        assert classifier.get_params() == {"engage_with_last": False}

        classifier.set_params(engage_with_last=True)
        assert classifier.get_params() == {"engage_with_last": True}

        with pytest.raises(InvalidArgumentError) as excinfo:
            classifier.set_params(helloworld=1.0)
        assert (
            str(excinfo.value) == "The given argument helloworld is not in the "
            "class 'PersistentClassifier'."
        )


@pytest.fixture
def train_cases():
    knowledges = [
        models.Knowledge({1: models.KnowledgeComponent(mean=0.27, variance=1e-9)}),
        models.Knowledge(
            {
                2: models.KnowledgeComponent(mean=0.13, variance=1e-9),
                3: models.KnowledgeComponent(mean=0.73, variance=1e-9),
            }
        ),
        models.Knowledge(
            {
                1: models.KnowledgeComponent(mean=0.24, variance=1e-9),
                3: models.KnowledgeComponent(mean=0.67, variance=1e-9),
            }
        ),
    ]
    times = [54, 1084, 53811]
    return (
        [
            models.EventModel(knowledge, time)
            for knowledge, time in zip(knowledges, times)
        ],
        [True, False, False],
    )


@pytest.fixture
def test_events():
    knowledges = [
        models.Knowledge(
            {
                1: models.KnowledgeComponent(mean=0.53, variance=1e-9),
                2: models.KnowledgeComponent(mean=0.28, variance=1e-9),
            }
        ),
        models.Knowledge(
            {
                2: models.KnowledgeComponent(mean=0.24, variance=1e-9),
                3: models.KnowledgeComponent(mean=0.61, variance=1e-9),
            }
        ),
        models.Knowledge(
            {
                1: models.KnowledgeComponent(mean=0.72, variance=1e-9),
                3: models.KnowledgeComponent(mean=0.53, variance=1e-9),
            }
        ),
    ]
    times = [54000, 215722, 479128]
    return [
        models.EventModel(knowledge, time) for knowledge, time in zip(knowledges, times)
    ]


class TestKnowledgeClassifier:
    def test_knowledge_positive_only_no_update(self, train_cases):
        classifier = learning.KnowledgeClassifier()

        train_events, _ = train_cases
        for event in train_events:
            classifier.fit(event, False)

        learner_model = classifier.get_learner_model()
        assert learner_model.number_of_engagements == 0
        assert learner_model.number_of_non_engagements == len(train_events)
        assert not list(learner_model.knowledge.knowledge_components())

    def test_knowledge_get_set_params(self):
        classifier = learning.KnowledgeClassifier(draw_proba_type="dynamic")

        params = (
            "beta",
            "def_var",
            "draw_proba_factor",
            "draw_proba_static",
            "draw_proba_type",
            "init_skill",
            "learner_model",
            "positive_only",
            "tau",
            "threshold",
        )
        assert params == tuple(classifier.get_params())

        classifier.set_params(tau=0.2)
        assert classifier.get_params()["tau"] == 0.2

        with pytest.raises(TrueLearnValueError) as excinfo:
            classifier.set_params(draw_proba_type="static")
            classifier.predict_proba(models.EventModel())
        assert (
            str(excinfo.value) == "When draw_proba_type is set to static, "
            "the draw_proba_static should not be None."
        )

    def test_knowledge_positive_easy(self):
        classifier = learning.KnowledgeClassifier(init_skill=0.0, def_var=0.5)

        knowledge = models.Knowledge(
            {1: models.KnowledgeComponent(mean=0.0, variance=0.5)}
        )
        event = models.EventModel(knowledge)

        assert classifier.predict_proba(event) == 0.5

    def test_knowledge_throw(self):
        with pytest.raises(TrueLearnTypeError) as excinfo:
            learning.KnowledgeClassifier(threshold=0)
        assert (
            str(excinfo.value) == "The threshold parameter of "
            "class 'KnowledgeClassifier' "
            "must be one of the classes in ['float']. Got 'int' instead."
        )

        with pytest.raises(TrueLearnTypeError) as excinfo:
            learning.KnowledgeClassifier(init_skill=0)
        assert (
            str(excinfo.value) == "The init_skill parameter of "
            "class 'KnowledgeClassifier' "
            "must be one of the classes in ['float']. Got 'int' instead."
        )

        with pytest.raises(TrueLearnTypeError) as excinfo:
            learning.KnowledgeClassifier(def_var=1)
        assert (
            str(excinfo.value) == "The def_var parameter of "
            "class 'KnowledgeClassifier' "
            "must be one of the classes in ['float']. Got 'int' instead."
        )

        with pytest.raises(TrueLearnTypeError) as excinfo:
            learning.KnowledgeClassifier(beta=0)
        assert (
            str(excinfo.value) == "The beta parameter of "
            "class 'KnowledgeClassifier' "
            "must be one of the classes in ['float']. Got 'int' instead."
        )

        with pytest.raises(TrueLearnTypeError) as excinfo:
            learning.KnowledgeClassifier(tau=1)
        assert (
            str(excinfo.value) == "The tau parameter of "
            "class 'KnowledgeClassifier' "
            "must be one of the classes in ['float']. Got 'int' instead."
        )

        with pytest.raises(TrueLearnValueError) as excinfo:
            learning.KnowledgeClassifier(draw_proba_type="hello world")
        assert (
            str(excinfo.value) == "The draw_proba_type parameter of "
            "class 'KnowledgeClassifier' "
            "must be one of the value inside tuple ('static', 'dynamic'). "
            "Got 'hello world' instead."
        )

        with pytest.raises(TrueLearnValueError) as excinfo:
            learning.KnowledgeClassifier(
                draw_proba_type="static", draw_proba_static=None
            )
        assert (
            str(excinfo.value)
            == "When draw_proba_type is set to static, the draw_proba_static should "
            "not be None."
        )

        with pytest.raises(TrueLearnTypeError) as excinfo:
            learning.KnowledgeClassifier(draw_proba_factor=1)
        assert (
            str(excinfo.value) == "The draw_proba_factor parameter of "
            "class 'KnowledgeClassifier' "
            "must be one of the classes in ['float']. "
            "Got 'int' instead."
        )

    def test_knowledge_classifier_with_disabled_positive_only(
        self, train_cases, test_events
    ):
        classifier = learning.KnowledgeClassifier(positive_only=False)

        train_events, train_labels = train_cases
        for event, label in zip(train_events, train_labels):
            classifier.fit(event, label)

        expected_results = [
            0.31931075120919905,
            0.032273232963724946,
            0.039341977637590585,
        ]
        actual_results = [classifier.predict_proba(event) for event in test_events]

        check_farray_close(actual_results, expected_results)

    def test_knowledge_classifier_with_draw_proba_static(
        self, train_cases, test_events
    ):
        classifier = learning.KnowledgeClassifier(
            draw_proba_type="static", draw_proba_static=0.3
        )

        train_events, train_labels = train_cases
        for event, label in zip(train_events, train_labels):
            classifier.fit(event, label)

        expected_results = [
            0.46850316736905456,
            0.1976625433589795,
            0.26525358732501003,
        ]
        actual_results = [classifier.predict_proba(event) for event in test_events]

        check_farray_close(actual_results, expected_results)


class TestNoveltyClassifier:
    def test_novelty_positive_only_no_update(self, train_cases):
        classifier = learning.NoveltyClassifier(positive_only=True)

        train_events, _ = train_cases
        for event in train_events:
            classifier.fit(event, False)

        learner_model = classifier.get_learner_model()
        assert learner_model.number_of_engagements == 0
        assert learner_model.number_of_non_engagements == len(train_events)
        assert not list(learner_model.knowledge.knowledge_components())

    def test_novelty_get_set_params(self):
        classifier = learning.NoveltyClassifier(draw_proba_type="dynamic")

        params = (
            "beta",
            "def_var",
            "draw_proba_factor",
            "draw_proba_static",
            "draw_proba_type",
            "init_skill",
            "learner_model",
            "positive_only",
            "tau",
            "threshold",
        )
        assert tuple(classifier.get_params()) == params

        classifier.set_params(tau=0.2)
        assert classifier.get_params()["tau"] == 0.2

        with pytest.raises(TrueLearnValueError) as excinfo:
            classifier.set_params(draw_proba_type="static")
            classifier.predict_proba(models.EventModel())
        assert (
            str(excinfo.value) == "When draw_proba_type is set to static, "
            "the draw_proba_static should not be None."
        )

    def test_novelty_positive_easy(self):
        classifier = learning.NoveltyClassifier(init_skill=0.0, def_var=0.5, beta=0.5)

        knowledge = models.Knowledge(
            {1: models.KnowledgeComponent(mean=0.0, variance=0.5)}
        )
        event = models.EventModel(knowledge)

        assert math.isclose(classifier.predict_proba(event), 0.5773502691896257)

    def test_novelty_throw(self):
        with pytest.raises(TrueLearnTypeError) as excinfo:
            learning.NoveltyClassifier(threshold=0)
        assert (
            str(excinfo.value) == "The threshold parameter of "
            "class 'NoveltyClassifier' "
            "must be one of the classes in ['float']. Got 'int' instead."
        )

        with pytest.raises(TrueLearnTypeError) as excinfo:
            learning.NoveltyClassifier(init_skill=0)
        assert (
            str(excinfo.value) == "The init_skill parameter of "
            "class 'NoveltyClassifier' "
            "must be one of the classes in ['float']. Got 'int' instead."
        )

        with pytest.raises(TrueLearnTypeError) as excinfo:
            learning.NoveltyClassifier(def_var=1)
        assert (
            str(excinfo.value) == "The def_var parameter of "
            "class 'NoveltyClassifier' "
            "must be one of the classes in ['float']. Got 'int' instead."
        )

        with pytest.raises(TrueLearnTypeError) as excinfo:
            learning.NoveltyClassifier(beta=0)
        assert (
            str(excinfo.value) == "The beta parameter of "
            "class 'NoveltyClassifier' "
            "must be one of the classes in ['float']. Got 'int' instead."
        )

        with pytest.raises(TrueLearnTypeError) as excinfo:
            learning.NoveltyClassifier(tau=1)
        assert (
            str(excinfo.value) == "The tau parameter of "
            "class 'NoveltyClassifier' "
            "must be one of the classes in ['float']. Got 'int' instead."
        )

        with pytest.raises(TrueLearnValueError) as excinfo:
            learning.NoveltyClassifier(draw_proba_type="hello world")
        assert (
            str(excinfo.value) == "The draw_proba_type parameter of "
            "class 'NoveltyClassifier' "
            "must be one of the value inside tuple ('static', 'dynamic'). "
            "Got 'hello world' instead."
        )

        with pytest.raises(TrueLearnValueError) as excinfo:
            learning.NoveltyClassifier(draw_proba_type="static", draw_proba_static=None)
        assert (
            str(excinfo.value)
            == "When draw_proba_type is set to static, the draw_proba_static should "
            "not be None."
        )

        with pytest.raises(TrueLearnTypeError) as excinfo:
            learning.NoveltyClassifier(draw_proba_factor=1)
        assert (
            str(excinfo.value) == "The draw_proba_factor parameter of "
            "class 'NoveltyClassifier' "
            "must be one of the classes in ['float']. Got 'int' instead."
        )

    def test_novelty_classifier_draw(self, train_cases, test_events):
        classifier = learning.NoveltyClassifier()

        train_events, train_labels = train_cases

        for event, label in zip(train_events, train_labels):
            classifier.fit(event, label)

        expected_results = [0.47693688479183843, 0.3006713231101619, 0.2443191280178933]
        actual_results = [classifier.predict_proba(event) for event in test_events]

        check_farray_close(actual_results, expected_results)

    def test_novelty_classifier_draws(self, train_cases, test_events):
        classifier = learning.NoveltyClassifier(positive_only=False)

        train_events, train_labels = train_cases
        # to test the case where the learner/content wins
        train_labels = [False, True, True]

        for event, label in zip(train_events, train_labels):
            classifier.fit(event, label)

        expected_results = [0.5734480552991874, 0.6807791229902742, 0.5291044653108932]
        actual_results = [classifier.predict_proba(event) for event in test_events]

        check_farray_close(actual_results, expected_results)

    def test_novelty_classifier_difference_zero(self):
        classifier = learning.NoveltyClassifier(
            init_skill=0.0, def_var=0.5, positive_only=False
        )

        train_events = [
            models.EventModel(
                models.Knowledge({1: models.KnowledgeComponent(mean=0.0, variance=0.5)})
            )
        ]

        # to test the case where the learner/content wins
        train_labels = [False]

        for event, label in zip(train_events, train_labels):
            classifier.fit(event, label)

        kc = list(classifier.get_learner_model().knowledge.knowledge_components())[0]
        assert kc.mean == 0.0
        assert math.isclose(kc.variance, 0.5)

    def test_novelty_classifier_difference_greater_than_zero(self):
        classifier = learning.NoveltyClassifier(
            init_skill=0.1, def_var=0.5, positive_only=False
        )

        train_events = [
            models.EventModel(
                models.Knowledge({1: models.KnowledgeComponent(mean=0.0, variance=0.5)})
            )
        ]

        # to test the case where the learner/content wins
        train_labels = [False]

        for event, label in zip(train_events, train_labels):
            classifier.fit(event, label)

        kc = list(classifier.get_learner_model().knowledge.knowledge_components())[0]
        assert math.isclose(kc.mean, 0.43237206984358406)
        assert math.isclose(kc.variance, 0.37618053129883994)


class TestInterestClassifier:
    def test_interest_get_set_params(self):
        classifier = learning.InterestClassifier(draw_proba_type="dynamic")

        params = (
            "beta",
            "decay_func_factor",
            "decay_func_type",
            "def_var",
            "draw_proba_factor",
            "draw_proba_static",
            "draw_proba_type",
            "init_skill",
            "learner_model",
            "positive_only",
            "tau",
            "threshold",
        )
        assert tuple(classifier.get_params()) == params

        classifier.set_params(tau=0.2)
        assert classifier.get_params()["tau"] == 0.2

        with pytest.raises(TrueLearnValueError) as excinfo:
            classifier.set_params(draw_proba_type="static")
            classifier.predict_proba(models.EventModel())
        assert (
            str(excinfo.value) == "When draw_proba_type is set to static, "
            "the draw_proba_static should not be None."
        )

    def test_interest_positive_easy(self):
        classifier = learning.InterestClassifier(init_skill=0.0, def_var=0.5)

        knowledge = models.Knowledge(
            {1: models.KnowledgeComponent(mean=0.0, variance=0.5)}
        )
        event = models.EventModel(knowledge)

        assert classifier.predict_proba(event) == 0.5

    def test_interest_throw(self):
        with pytest.raises(TrueLearnTypeError) as excinfo:
            learning.InterestClassifier(threshold=0)
        assert (
            str(excinfo.value) == "The threshold parameter of "
            "class 'InterestClassifier' "
            "must be one of the classes in ['float']. Got 'int' instead."
        )

        with pytest.raises(TrueLearnTypeError) as excinfo:
            learning.InterestClassifier(init_skill=0)
        assert (
            str(excinfo.value) == "The init_skill parameter of "
            "class 'InterestClassifier' "
            "must be one of the classes in ['float']. Got 'int' instead."
        )

        with pytest.raises(TrueLearnTypeError) as excinfo:
            learning.InterestClassifier(def_var=1)
        assert (
            str(excinfo.value) == "The def_var parameter of "
            "class 'InterestClassifier' "
            "must be one of the classes in ['float']. Got 'int' instead."
        )

        with pytest.raises(TrueLearnTypeError) as excinfo:
            learning.InterestClassifier(beta=0)
        assert (
            str(excinfo.value) == "The beta parameter of "
            "class 'InterestClassifier' "
            "must be one of the classes in ['float']. Got 'int' instead."
        )

        with pytest.raises(TrueLearnTypeError) as excinfo:
            learning.InterestClassifier(tau=1)
        assert (
            str(excinfo.value) == "The tau parameter of "
            "class 'InterestClassifier' "
            "must be one of the classes in ['float']. Got 'int' instead."
        )

        with pytest.raises(TrueLearnValueError) as excinfo:
            learning.InterestClassifier(draw_proba_type="hello world")
        assert (
            str(excinfo.value) == "The draw_proba_type parameter of "
            "class 'InterestClassifier' "
            "must be one of the value inside tuple ('static', 'dynamic'). "
            "Got 'hello world' instead."
        )

        with pytest.raises(TrueLearnValueError) as excinfo:
            learning.InterestClassifier(
                draw_proba_type="static", draw_proba_static=None
            )
        assert (
            str(excinfo.value)
            == "When draw_proba_type is set to static, the draw_proba_static should "
            "not be None."
        )

        with pytest.raises(TrueLearnTypeError) as excinfo:
            learning.InterestClassifier(draw_proba_factor=1)
        assert (
            str(excinfo.value) == "The draw_proba_factor parameter of "
            "class 'InterestClassifier' "
            "must be one of the classes in ['float']. Got 'int' instead."
        )

        with pytest.raises(TrueLearnValueError) as excinfo:
            learning.InterestClassifier(decay_func_type="hello world")
        assert (
            str(excinfo.value) == "The decay_func_type parameter of "
            "class 'InterestClassifier' "
            "must be one of the value inside tuple ('short', 'long'). "
            "Got 'hello world' instead."
        )

        with pytest.raises(TrueLearnTypeError) as excinfo:
            learning.InterestClassifier(decay_func_factor=1)
        assert (
            str(excinfo.value) == "The decay_func_factor parameter of "
            "class 'InterestClassifier' "
            "must be one of the classes in ['float']. Got 'int' instead."
        )

        with pytest.raises(TrueLearnValueError) as excinfo:
            classifier = learning.InterestClassifier()
            classifier.fit(models.EventModel(), True)
        assert (
            str(excinfo.value)
            == "The event time should not be None when using InterestClassifier."
        )

        with pytest.raises(TrueLearnValueError) as excinfo:
            learner_model = models.LearnerModel(
                models.Knowledge(
                    {1: models.KnowledgeComponent(mean=0.0, variance=1.0)}
                ),
            )
            classifier = learning.InterestClassifier(learner_model=learner_model)
            classifier.fit(
                models.EventModel(
                    models.Knowledge(
                        {1: models.KnowledgeComponent(mean=0.0, variance=1.0)}
                    ),
                    event_time=0.0,
                ),
                True,
            )
        assert (
            str(excinfo.value) == "The timestamp field of knowledge component"
            " should not be None if using InterestClassifier."
        )

    def test_interest_classifier(self, train_cases, test_events):
        classifier = learning.InterestClassifier()

        train_events, train_labels = train_cases
        for event, label in zip(train_events, train_labels):
            classifier.fit(event, label)

        expected_results = [0.8648794445446283, 0.8438279621999456, 0.7777471206958368]
        actual_results = [classifier.predict_proba(event) for event in test_events]

        check_farray_close(actual_results, expected_results)

    def test_interest_classifier_decay_func_type(self, train_cases, test_events):
        classifier_short = learning.InterestClassifier(decay_func_type="short")
        classifier_long = learning.InterestClassifier(decay_func_type="long")
        train_events, train_labels = train_cases
        for event, label in zip(train_events, train_labels):
            classifier_short.fit(event, label)
            classifier_long.fit(event, label)

        # should be the same because factor is 0
        assert [classifier_short.predict_proba(event) for event in test_events] == [
            classifier_long.predict_proba(event) for event in test_events
        ]


class TestINKClassifier:
    def test_ink_default_value(self):
        classifier = learning.INKClassifier()

        (novelty_model, interest_model, meta_weights) = classifier.get_learner_model()
        assert (
            meta_weights.interest_weights
            == meta_weights.novelty_weights
            == models.LearnerMetaWeights.Weights(0.5, 0.5)
        )
        assert meta_weights.bias_weights == models.LearnerMetaWeights.Weights(0.0, 0.5)
        assert not list(novelty_model.knowledge.knowledge_components()) and not list(
            interest_model.knowledge.knowledge_components()
        )

    def test_ink_classifier_customize(self, train_cases, test_events):
        learner_meta_weights = models.LearnerMetaWeights(
            bias_weights=models.LearnerMetaWeights.Weights(0.1, 0.5)
        )
        novelty_classifier = learning.NoveltyClassifier(def_var=0.4)
        interest_classifier = learning.InterestClassifier(beta=0.2)
        classifier = learning.INKClassifier(
            learner_meta_weights=learner_meta_weights,
            novelty_classifier=novelty_classifier,
            interest_classifier=interest_classifier,
        )

        train_events, train_labels = train_cases
        for event, label in zip(train_events, train_labels):
            classifier.fit(event, label)

        expected_results = [0.40575267541878457, 0.36519542301026875, 0.33362493980730495]
        actual_results = [classifier.predict_proba(event) for event in test_events]

        check_farray_close(actual_results, expected_results)

    def test_ink_get_set_params(self):
        classifier = learning.INKClassifier()

        params = (
            "greedy",
            "interest_classifier",
            "learner_meta_weights",
            "novelty_classifier",
            "tau",
            "threshold",
        )
        assert tuple(classifier.get_params(deep=False)) == params

        # set simple parameter
        classifier.set_params(greedy=True)
        assert classifier.get_params()["greedy"]

        # set nested parameter
        classifier.set_params(novelty_classifier__tau=0.2)
        assert classifier.get_params()["novelty_classifier__tau"] == 0.2

    def test_ink_throw(self):
        with pytest.raises(TrueLearnTypeError) as excinfo:
            learning.INKClassifier(threshold=0)
        assert (
            str(excinfo.value) == "The threshold parameter of "
            "class 'INKClassifier' "
            "must be one of the classes in ['float']. Got 'int' instead."
        )

        with pytest.raises(TrueLearnTypeError) as excinfo:
            learning.INKClassifier(tau=1)
        assert (
            str(excinfo.value) == "The tau parameter of "
            "class 'INKClassifier' "
            "must be one of the classes in ['float']. Got 'int' instead."
        )

    def test_ink_classifier(self, train_cases, test_events):
        classifier = learning.INKClassifier()

        train_events, train_labels = train_cases
        for event, label in zip(train_events, train_labels):
            classifier.fit(event, label)

        expected_results = [0.3844070661899784, 0.3398805698754434, 0.3133264788862059]
        actual_results = [classifier.predict_proba(event) for event in test_events]

        check_farray_close(actual_results, expected_results)

    def test_ink_classifier_greedy(self):
        classifier = learning.INKClassifier(greedy=True)

        train_events = [
            models.EventModel(
                models.Knowledge(
                    {1: models.KnowledgeComponent(mean=0.0, variance=0.5, timestamp=0)}
                ),
                event_time=0,
            )
        ]
        train_labels = [True]
        for event, label in zip(train_events, train_labels):
            classifier.fit(event, label)

        # if the classifier is in greedy mode and the prediction is correct
        # the classifier will not adjust its weights
        (_, _, meta_weights) = classifier.get_learner_model()
        assert (
            meta_weights.novelty_weights
            == meta_weights.interest_weights
            == models.LearnerMetaWeights.Weights(0.5, 0.5)
        )
        assert meta_weights.bias_weights == models.LearnerMetaWeights.Weights(0.0, 0.5)
