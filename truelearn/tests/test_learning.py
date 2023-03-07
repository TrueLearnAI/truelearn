# pylint: disable=missing-function-docstring,missing-class-docstring,protected-access
import math

import pytest

from truelearn import learning, models
from ..learning import _base


class MockClassifier(_base.BaseClassifier):
    _parameter_constraints = {
        **_base.BaseClassifier._parameter_constraints,
        "holder": int,
    }

    def __init__(self):  # noqa: D107
        self.holder = 42

    def fit(self, _x, _y):
        ...

    def predict(self, _x):
        ...

    def predict_proba(self, _x):
        ...


class TestBase:
    def test_useless_key_with_get_params_throw(self, monkeypatch):
        monkeypatch.setattr(
            MockClassifier,
            "_parameter_constraints",
            {**MockClassifier._parameter_constraints, "useless_key": type(None)},
        )

        classifier = MockClassifier()
        with pytest.raises(ValueError) as excinfo:
            classifier.get_params()
        assert (
            "The specified parameter name useless_key is not in the "
            "<class 'truelearn.tests.test_learning.MockClassifier'>."
            == str(excinfo.value)
        )

    def test_set_params_empty_no_effect(self):
        classifier = MockClassifier()
        params = classifier.get_params()
        classifier.set_params()

        assert params == classifier.get_params() == {"holder": classifier.holder}

    def test_useless_key_with_validate_params_throw(self, monkeypatch):
        monkeypatch.setattr(
            MockClassifier,
            "_parameter_constraints",
            {**MockClassifier._parameter_constraints, "useless_key": type(None)},
        )

        classifier = MockClassifier()
        with pytest.raises(ValueError) as excinfo:
            classifier._validate_params()
        assert (
            "The specified parameter name useless_key is not in the "
            "<class 'truelearn.tests.test_learning.MockClassifier'>."
            == str(excinfo.value)
        )

    def test_validate_params_type_mismatch_throw(self, monkeypatch):
        parameter_constraints = {**MockClassifier._parameter_constraints,
                                 "holder": [float, str, type(None)]}

        monkeypatch.setattr(
            MockClassifier,
            "_parameter_constraints",
            parameter_constraints,
        )

        classifier = MockClassifier()
        with pytest.raises(TypeError) as excinfo:
            classifier._validate_params()
        assert (
            "The holder parameter of "
            "<class 'truelearn.tests.test_learning.MockClassifier'> __init__"
            " function must be one of the classes in ['float', 'str', 'NoneType']. "
            "Got <class 'int'> instead." == str(excinfo.value)
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
        assert classifier.predict_proba(models.EventModel()) == 0.0

    def test_majority_get_set_params(self):
        classifier = learning.MajorityClassifier()

        assert classifier.get_params() == {
            "engagement": 0,
            "non_engagement": 0,
        }

        classifier.set_params(engagement=1)
        assert classifier.get_params() == {"engagement": 1, "non_engagement": 0}

        with pytest.raises(TypeError) as excinfo:
            classifier.set_params(non_engagement=1.0)
        assert (
            "The non_engagement parameter of "
            "<class 'truelearn.learning._majority_classifier.MajorityClassifier'> "
            "must be <class 'int'>. Got <class 'float'> instead." == str(excinfo.value)
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

        with pytest.raises(KeyError) as excinfo:
            classifier.set_params(helloworld=1.0)
        assert (
            '"'
            "The given parameter helloworld is not in the "
            "<class 'truelearn.learning._persistent_classifier.PersistentClassifier'>."
            '"' == str(excinfo.value)
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

        with pytest.raises(ValueError) as excinfo:
            classifier.set_params(draw_proba_type="static")
            classifier.predict_proba(models.EventModel())
        assert (
            "When draw_proba_type is set to static, "
            "the draw_proba_static should not be None." == str(excinfo.value)
        )

    def test_knowledge_positive_easy(self):
        classifier = learning.KnowledgeClassifier(init_skill=0.0, def_var=0.5)

        knowledge = models.Knowledge(
            {1: models.KnowledgeComponent(mean=0.0, variance=0.5)}
        )
        event = models.EventModel(knowledge)

        assert classifier.predict_proba(event) == 0.5

    def test_knowledge_throw(self):
        with pytest.raises(TypeError) as excinfo:
            learning.KnowledgeClassifier(threshold=0)
        assert (
            "The threshold parameter of "
            "<class 'truelearn.learning._knowledge_classifier.KnowledgeClassifier'> "
            "must be <class 'float'>. Got <class 'int'> instead." == str(excinfo.value)
        )

        with pytest.raises(TypeError) as excinfo:
            learning.KnowledgeClassifier(init_skill=0)
        assert (
            "The init_skill parameter of "
            "<class 'truelearn.learning._knowledge_classifier.KnowledgeClassifier'> "
            "must be <class 'float'>. Got <class 'int'> instead." == str(excinfo.value)
        )

        with pytest.raises(TypeError) as excinfo:
            learning.KnowledgeClassifier(def_var=1)
        assert (
            "The def_var parameter of "
            "<class 'truelearn.learning._knowledge_classifier.KnowledgeClassifier'> "
            "must be <class 'float'>. Got <class 'int'> instead." == str(excinfo.value)
        )

        with pytest.raises(TypeError) as excinfo:
            learning.KnowledgeClassifier(beta=0)
        assert (
            "The beta parameter of "
            "<class 'truelearn.learning._knowledge_classifier.KnowledgeClassifier'> "
            "must be <class 'float'>. Got <class 'int'> instead." == str(excinfo.value)
        )

        with pytest.raises(TypeError) as excinfo:
            learning.KnowledgeClassifier(tau=1)
        assert (
            "The tau parameter of "
            "<class 'truelearn.learning._knowledge_classifier.KnowledgeClassifier'> "
            "must be <class 'float'>. Got <class 'int'> instead." == str(excinfo.value)
        )

        with pytest.raises(ValueError) as excinfo:
            learning.KnowledgeClassifier(draw_proba_type="hello world")
        assert (
            "The draw_proba_type parameter of "
            "<class 'truelearn.learning._knowledge_classifier.KnowledgeClassifier'> "
            "must be one of the value inside tuple ('static', 'dynamic'). "
            "Got 'hello world' instead." == str(excinfo.value)
        )

        with pytest.raises(ValueError) as excinfo:
            learning.KnowledgeClassifier(
                draw_proba_type="static", draw_proba_static=None
            )
        assert (
            "When draw_proba_type is set to static, the draw_proba_static should not be"
            " None." == str(excinfo.value)
        )

        with pytest.raises(TypeError) as excinfo:
            learning.KnowledgeClassifier(draw_proba_factor=1)
        assert (
            "The draw_proba_factor parameter of "
            "<class 'truelearn.learning._knowledge_classifier.KnowledgeClassifier'> "
            "must be <class 'float'>. Got <class 'int'> instead." == str(excinfo.value)
        )

    def test_knowledge_classifier_with_disabled_positive_only(
        self, train_cases, test_events
    ):
        classifier = learning.KnowledgeClassifier(positive_only=False)

        train_events, train_labels = train_cases
        for event, label in zip(train_events, train_labels):
            classifier.fit(event, label)

        expected_results = [
            0.31822146219026654,
            0.04616731561824022,
            0.0644849610860269,
        ]
        actual_results = [classifier.predict_proba(event) for event in test_events]

        assert expected_results == actual_results

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
            0.4632708833103084,
            0.20884274880379491,
            0.27614902348099163,
        ]
        actual_results = [classifier.predict_proba(event) for event in test_events]

        assert expected_results == actual_results


class TestNoveltyClassifier:
    def test_novelty_positive_only_no_update(self, train_cases):
        classifier = learning.NoveltyClassifier()

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
        assert params == tuple(classifier.get_params())

        classifier.set_params(tau=0.2)
        assert classifier.get_params()["tau"] == 0.2

        with pytest.raises(ValueError) as excinfo:
            classifier.set_params(draw_proba_type="static")
            classifier.predict_proba(models.EventModel())
        assert (
            "When draw_proba_type is set to static, "
            "the draw_proba_static should not be None." == str(excinfo.value)
        )

    def test_novelty_positive_easy(self):
        classifier = learning.NoveltyClassifier(init_skill=0.0, def_var=0.5, beta=0.5)

        knowledge = models.Knowledge(
            {1: models.KnowledgeComponent(mean=0.0, variance=0.5)}
        )
        event = models.EventModel(knowledge)

        assert classifier.predict_proba(event) == 0.5773502691896257

    def test_novelty_throw(self):
        with pytest.raises(TypeError) as excinfo:
            learning.NoveltyClassifier(threshold=0)
        assert (
            "The threshold parameter of "
            "<class 'truelearn.learning._novelty_classifier.NoveltyClassifier'> "
            "must be <class 'float'>. Got <class 'int'> instead." == str(excinfo.value)
        )

        with pytest.raises(TypeError) as excinfo:
            learning.NoveltyClassifier(init_skill=0)
        assert (
            "The init_skill parameter of "
            "<class 'truelearn.learning._novelty_classifier.NoveltyClassifier'> "
            "must be <class 'float'>. Got <class 'int'> instead." == str(excinfo.value)
        )

        with pytest.raises(TypeError) as excinfo:
            learning.NoveltyClassifier(def_var=1)
        assert (
            "The def_var parameter of "
            "<class 'truelearn.learning._novelty_classifier.NoveltyClassifier'> "
            "must be <class 'float'>. Got <class 'int'> instead." == str(excinfo.value)
        )

        with pytest.raises(TypeError) as excinfo:
            learning.NoveltyClassifier(beta=0)
        assert (
            "The beta parameter of "
            "<class 'truelearn.learning._novelty_classifier.NoveltyClassifier'> "
            "must be <class 'float'>. Got <class 'int'> instead." == str(excinfo.value)
        )

        with pytest.raises(TypeError) as excinfo:
            learning.NoveltyClassifier(tau=1)
        assert (
            "The tau parameter of "
            "<class 'truelearn.learning._novelty_classifier.NoveltyClassifier'> "
            "must be <class 'float'>. Got <class 'int'> instead." == str(excinfo.value)
        )

        with pytest.raises(ValueError) as excinfo:
            learning.NoveltyClassifier(draw_proba_type="hello world")
        assert (
            "The draw_proba_type parameter of "
            "<class 'truelearn.learning._novelty_classifier.NoveltyClassifier'> "
            "must be one of the value inside tuple ('static', 'dynamic'). "
            "Got 'hello world' instead." == str(excinfo.value)
        )

        with pytest.raises(ValueError) as excinfo:
            learning.NoveltyClassifier(draw_proba_type="static", draw_proba_static=None)
        assert (
            "When draw_proba_type is set to static, the draw_proba_static should not be"
            " None." == str(excinfo.value)
        )

        with pytest.raises(TypeError) as excinfo:
            learning.NoveltyClassifier(draw_proba_factor=1)
        assert (
            "The draw_proba_factor parameter of "
            "<class 'truelearn.learning._novelty_classifier.NoveltyClassifier'> "
            "must be <class 'float'>. Got <class 'int'> instead." == str(excinfo.value)
        )

    def test_novelty_classifier_draw(self, train_cases, test_events):
        classifier = learning.NoveltyClassifier()

        train_events, train_labels = train_cases

        for event, label in zip(train_events, train_labels):
            classifier.fit(event, label)

        expected_results = [
            0.20232035389852276,
            0.1385666227211696,
            0.11103582174277878,
        ]
        actual_results = [classifier.predict_proba(event) for event in test_events]

        assert expected_results == actual_results

    def test_novelty_classifier_draws(self, train_cases, test_events):
        classifier = learning.NoveltyClassifier(positive_only=False)

        train_events, train_labels = train_cases
        # to test the case where the learner/content wins
        train_labels = [False, True, True]

        for event, label in zip(train_events, train_labels):
            classifier.fit(event, label)

        expected_results = [
            0.24498763833059276,
            0.2509683717988391,
            0.26190237284122425,
        ]
        actual_results = [classifier.predict_proba(event) for event in test_events]

        assert expected_results == actual_results

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
        assert kc.mean == 0.4683333154754179
        assert kc.variance == 0.3562680695898126


class TestInterestClassifier:
    def test_interest_positive_only_no_update(self, train_cases):
        classifier = learning.InterestClassifier()

        train_events, _ = train_cases
        for event in train_events:
            classifier.fit(event, False)

        learner_model = classifier.get_learner_model()
        assert learner_model.number_of_engagements == 0
        assert learner_model.number_of_non_engagements == len(train_events)
        assert not list(learner_model.knowledge.knowledge_components())

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
        assert params == tuple(classifier.get_params())

        classifier.set_params(tau=0.2)
        assert classifier.get_params()["tau"] == 0.2

        with pytest.raises(ValueError) as excinfo:
            classifier.set_params(draw_proba_type="static")
            classifier.predict_proba(models.EventModel())
        assert (
            "When draw_proba_type is set to static, "
            "the draw_proba_static should not be None." == str(excinfo.value)
        )

    def test_interest_positive_easy(self):
        classifier = learning.InterestClassifier(init_skill=0.0, def_var=0.5)

        knowledge = models.Knowledge(
            {1: models.KnowledgeComponent(mean=0.0, variance=0.5)}
        )
        event = models.EventModel(knowledge)

        assert classifier.predict_proba(event) == 0.5

    def test_interest_throw(self):
        with pytest.raises(TypeError) as excinfo:
            learning.InterestClassifier(threshold=0)
        assert (
            "The threshold parameter of "
            "<class 'truelearn.learning._interest_classifier.InterestClassifier'> "
            "must be <class 'float'>. Got <class 'int'> instead." == str(excinfo.value)
        )

        with pytest.raises(TypeError) as excinfo:
            learning.InterestClassifier(init_skill=0)
        assert (
            "The init_skill parameter of "
            "<class 'truelearn.learning._interest_classifier.InterestClassifier'> "
            "must be <class 'float'>. Got <class 'int'> instead." == str(excinfo.value)
        )

        with pytest.raises(TypeError) as excinfo:
            learning.InterestClassifier(def_var=1)
        assert (
            "The def_var parameter of "
            "<class 'truelearn.learning._interest_classifier.InterestClassifier'> "
            "must be <class 'float'>. Got <class 'int'> instead." == str(excinfo.value)
        )

        with pytest.raises(TypeError) as excinfo:
            learning.InterestClassifier(beta=0)
        assert (
            "The beta parameter of "
            "<class 'truelearn.learning._interest_classifier.InterestClassifier'> "
            "must be <class 'float'>. Got <class 'int'> instead." == str(excinfo.value)
        )

        with pytest.raises(TypeError) as excinfo:
            learning.InterestClassifier(tau=1)
        assert (
            "The tau parameter of "
            "<class 'truelearn.learning._interest_classifier.InterestClassifier'> "
            "must be <class 'float'>. Got <class 'int'> instead." == str(excinfo.value)
        )

        with pytest.raises(ValueError) as excinfo:
            learning.InterestClassifier(draw_proba_type="hello world")
        assert (
            "The draw_proba_type parameter of "
            "<class 'truelearn.learning._interest_classifier.InterestClassifier'> "
            "must be one of the value inside tuple ('static', 'dynamic'). "
            "Got 'hello world' instead." == str(excinfo.value)
        )

        with pytest.raises(ValueError) as excinfo:
            learning.InterestClassifier(
                draw_proba_type="static", draw_proba_static=None
            )
        assert (
            "When draw_proba_type is set to static, the draw_proba_static should not be"
            " None." == str(excinfo.value)
        )

        with pytest.raises(TypeError) as excinfo:
            learning.InterestClassifier(draw_proba_factor=1)
        assert (
            "The draw_proba_factor parameter of "
            "<class 'truelearn.learning._interest_classifier.InterestClassifier'> "
            "must be <class 'float'>. Got <class 'int'> instead." == str(excinfo.value)
        )

        with pytest.raises(ValueError) as excinfo:
            learning.InterestClassifier(decay_func_type="hello world")
        assert (
            "The decay_func_type parameter of "
            "<class 'truelearn.learning._interest_classifier.InterestClassifier'> "
            "must be one of the value inside tuple ('short', 'long'). "
            "Got 'hello world' instead." == str(excinfo.value)
        )

        with pytest.raises(TypeError) as excinfo:
            learning.InterestClassifier(decay_func_factor=1)
        assert (
            "The decay_func_factor parameter of "
            "<class 'truelearn.learning._interest_classifier.InterestClassifier'> "
            "must be <class 'float'>. Got <class 'int'> instead." == str(excinfo.value)
        )

        with pytest.raises(ValueError) as excinfo:
            classifier = learning.InterestClassifier()
            classifier.fit(models.EventModel(), True)
        assert (
            "The event time should not be None when using InterestClassifier."
            == str(excinfo.value)
        )

        with pytest.raises(ValueError) as excinfo:
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
            "The timestamp field of knowledge component"
            " should not be None if using InterestClassifier." == str(excinfo.value)
        )

    def test_interest_classifier(self, train_cases, test_events):
        classifier = learning.InterestClassifier(positive_only=False)

        train_events, train_labels = train_cases
        for event, label in zip(train_events, train_labels):
            classifier.fit(event, label)

        expected_results = [0.8253463535369645, 0.8124833431982229, 0.7434285783213926]
        actual_results = [classifier.predict_proba(event) for event in test_events]

        assert expected_results == actual_results

    def test_interest_classifier_decay_func_type(self, train_cases, test_events):
        classifier_short = learning.InterestClassifier(
            positive_only=False, decay_func_type="short"
        )
        classifier_long = learning.InterestClassifier(
            positive_only=False, decay_func_type="long"
        )
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

        learner_model = classifier.get_learner_model()
        assert (
            learner_model.bias_weight
            == learner_model.interest_weight
            == learner_model.novelty_weight
            == models.LearnerMetaModel.Weights()
        )
        assert not list(
            learner_model.learner_novelty.knowledge.knowledge_components()
        ) and not list(learner_model.learner_interest.knowledge.knowledge_components())

    def test_ink_get_set_params(self):
        classifier = learning.INKClassifier()

        params = (
            "bias_weights",
            "greedy",
            "interest_classifier",
            "interest_weights",
            "novelty_classifier",
            "novelty_weights",
            "tau",
            "threshold",
        )
        assert params == tuple(classifier.get_params(deep=False))

        # set simple parameter
        classifier.set_params(greedy=True)
        assert classifier.get_params()["greedy"]

        # set nested parameter
        classifier.set_params(novelty_classifier__tau=0.2)
        assert classifier.get_params()["novelty_classifier__tau"] == 0.2

    def test_ink_throw(self):
        with pytest.raises(TypeError) as excinfo:
            learning.INKClassifier(threshold=0)
        assert (
            "The threshold parameter of "
            "<class 'truelearn.learning._ink_classifier.INKClassifier'> "
            "must be <class 'float'>. Got <class 'int'> instead." == str(excinfo.value)
        )

        with pytest.raises(TypeError) as excinfo:
            learning.INKClassifier(tau=1)
        assert (
            "The tau parameter of "
            "<class 'truelearn.learning._ink_classifier.INKClassifier'> "
            "must be <class 'float'>. Got <class 'int'> instead." == str(excinfo.value)
        )

    def test_ink_classifier(self, train_cases, test_events):
        classifier = learning.INKClassifier()

        train_events, train_labels = train_cases
        for event, label in zip(train_events, train_labels):
            classifier.fit(event, label)

        expected_results = [
            0.24337755209294626,
            0.21257650005484793,
            0.2160900839287269,
        ]
        actual_results = [classifier.predict_proba(event) for event in test_events]

        assert expected_results == actual_results

    def test_ink_classifier_customize(self, train_cases, test_events):
        novelty_classifier = learning.NoveltyClassifier()
        interest_classifier = learning.InterestClassifier()
        novelty_weights = models.LearnerMetaModel.Weights()
        interest_weights = models.LearnerMetaModel.Weights()
        bias_weights = models.LearnerMetaModel.Weights()

        classifier = learning.INKClassifier(
            novelty_classifier=novelty_classifier,
            interest_classifier=interest_classifier,
            novelty_weights=novelty_weights,
            interest_weights=interest_weights,
            bias_weights=bias_weights,
        )

        train_events, train_labels = train_cases
        for event, label in zip(train_events, train_labels):
            classifier.fit(event, label)

        expected_results = [
            0.24337755209294626,
            0.21257650005484793,
            0.2160900839287269,
        ]
        actual_results = [classifier.predict_proba(event) for event in test_events]

        assert expected_results == actual_results

    def test_ink_classifier_greedy(self):
        classifier = learning.INKClassifier(greedy=True)

        train_events = [
            models.EventModel(
                models.Knowledge({1: models.KnowledgeComponent(mean=0.0, variance=0.5)})
            )
        ]
        train_labels = [False]
        for event, label in zip(train_events, train_labels):
            classifier.fit(event, label)

        # if the classifier is in greedy mode and the prediction is correct
        # the classifier will not adjust its weights
        assert (
            classifier.novelty_weights
            == classifier.interest_weights
            == classifier.bias_weights
            == models.LearnerMetaModel.Weights(0.0, 0.5)
        )
