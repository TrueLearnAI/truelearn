# pylint: disable=missing-function-docstring,missing-class-docstring
import pytest

from truelearn import learning, models


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
        classifier = learning.KnowledgeClassifier()

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

    def test_knowledge_classifier(self, train_cases, test_events):
        classifier = learning.KnowledgeClassifier()

        train_events, train_labels = train_cases
        for event, label in zip(train_events, train_labels):
            classifier.fit(event, label)

        expected_results = [
            0.4616704074923047,
            0.20884274880379491,
            0.27487026385771485,
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
        classifier = learning.NoveltyClassifier(beta=0.5)

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

    def test_novelty_classifier_wins(self, train_cases, test_events):
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
        classifier = learning.InterestClassifier()

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

    def test_interest_classifier(self, train_cases, test_events):
        classifier = learning.InterestClassifier(positive_only=False)

        train_events, train_labels = train_cases
        for event, label in zip(train_events, train_labels):
            classifier.fit(event, label)

        expected_results = [0.8253463535369645, 0.8124833431982229, 0.7434285783213926]
        actual_results = [classifier.predict_proba(event) for event in test_events]

        assert expected_results == actual_results


class TestINKClassifier:
    def test_ink_default_value(self):
        classifier = learning.INKClassifier()

        learner_model = classifier.get_learner_model()
        assert (
            learner_model.bias_weight
            == learner_model.interest_weight
            == learner_model.novelty_weight
            == {"mean": 0.0, "variance": 0.5}
        )
        assert not list(
            learner_model.learner_novelty.knowledge.knowledge_components()
        ) and not list(learner_model.learner_interest.knowledge.knowledge_components())

    def test_ink_get_set_params(self):
        classifier = learning.INKClassifier()

        params = (
            "bias_weight",
            "greedy",
            "interest_classifier",
            "interest_weight",
            "novelty_classifier",
            "novelty_weight",
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

        with pytest.raises(ValueError) as excinfo:
            learning.INKClassifier(novelty_weight={})
        assert (
            'The novelty_weight must contain "mean" and "variance". Got {} instead.'
            == str(excinfo.value)
        )

        with pytest.raises(ValueError) as excinfo:
            learning.INKClassifier(interest_weight={})
        assert (
            'The interest_weight must contain "mean" and "variance". Got {} instead.'
            == str(excinfo.value)
        )

        with pytest.raises(ValueError) as excinfo:
            learning.INKClassifier(bias_weight={})
        assert (
            'The bias_weight must contain "mean" and "variance". Got {} instead.'
            == str(excinfo.value)
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
