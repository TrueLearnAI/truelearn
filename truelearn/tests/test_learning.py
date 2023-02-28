# pylint: disable=missing-function-docstring
import pytest

from truelearn import learning, models


def test_engage_classifier():
    classifier = learning.EngageClassifier()

    # no exception should occur
    classifier.fit(models.EventModel(), True)

    assert classifier.predict(models.EventModel())
    assert not classifier.get_params()
    assert classifier.predict_proba(models.EventModel()) == 1.0


def test_majority_classifier():
    classifier = learning.MajorityClassifier()

    # no exception should occur
    classifier.fit(models.EventModel(), True)

    assert classifier.predict(models.EventModel())
    assert classifier.get_params() == {
        "engagement": 1,
        "non_engagement": 0,
    }
    assert classifier.predict_proba(models.EventModel()) == 1.0

    classifier.fit(models.EventModel(), False)

    # because engagement == non_engagement
    assert not classifier.predict(models.EventModel())
    assert classifier.predict_proba(models.EventModel()) == 0.0


def test_persistent_classifier():
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


def test_knowledge_positive_only_no_update(train_cases):
    classifier = learning.KnowledgeClassifier()

    train_events, _ = train_cases
    for event in train_events:
        classifier.fit(event, False)

    learner_model = classifier.get_learner_model()
    assert learner_model.number_of_engagements == 0
    assert learner_model.number_of_non_engagements == len(train_events)
    assert not list(learner_model.knowledge.knowledge_components())


def test_knowledge_positive_easy():
    classifier = learning.KnowledgeClassifier()

    knowledge = models.Knowledge({1: models.KnowledgeComponent(mean=0.0, variance=0.5)})
    event = models.EventModel(knowledge)

    assert classifier.predict_proba(event) == 0.5


def test_knowledge_throw():
    with pytest.raises(TypeError) as excinfo:
        learning.KnowledgeClassifier(threshold=0)
    assert (
        "The threshold parameter of KnowledgeClassifier must be float. "
        "Got int instead." == str(excinfo.value)
    )

    with pytest.raises(TypeError) as excinfo:
        learning.KnowledgeClassifier(init_skill=0)
    assert (
        "The init_skill parameter of KnowledgeClassifier must be float. "
        "Got int instead." == str(excinfo.value)
    )

    with pytest.raises(TypeError) as excinfo:
        learning.KnowledgeClassifier(def_var=1)
    assert (
        "The def_var parameter of KnowledgeClassifier must be float. "
        "Got int instead." == str(excinfo.value)
    )

    with pytest.raises(TypeError) as excinfo:
        learning.KnowledgeClassifier(beta=0)
    assert (
        "The beta parameter of KnowledgeClassifier must be float. "
        "Got int instead." == str(excinfo.value)
    )

    with pytest.raises(TypeError) as excinfo:
        learning.KnowledgeClassifier(tau=1)
    assert (
        "The tau parameter of KnowledgeClassifier must be float. "
        "Got int instead." == str(excinfo.value)
    )

    with pytest.raises(ValueError) as excinfo:
        learning.KnowledgeClassifier(draw_proba_type="hello world")
    assert (
        "The draw_proba_type should be either static or dynamic. "
        "Got hello world instead." == str(excinfo.value)
    )

    with pytest.raises(ValueError) as excinfo:
        learning.KnowledgeClassifier(draw_proba_type="static", draw_proba_static=None)
    assert (
        "When draw_proba_type is set to static, the draw_proba_static should not be "
        "None." == str(excinfo.value)
    )

    with pytest.raises(TypeError) as excinfo:
        learning.KnowledgeClassifier(draw_proba_factor=1)
    assert (
        "The draw_proba_factor parameter of KnowledgeClassifier must be float. "
        "Got int instead." == str(excinfo.value)
    )


def test_knowledge_classifier(train_cases, test_events):
    classifier = learning.KnowledgeClassifier()

    train_events, train_labels = train_cases
    for event, label in zip(train_events, train_labels):
        classifier.fit(event, label)

    expected_results = [0.4616704074923047, 0.20884274880379491, 0.27487026385771485]
    actual_results = [classifier.predict_proba(event) for event in test_events]

    assert expected_results == actual_results


def test_novelty_classifier(train_cases, test_events):
    classifier = learning.NoveltyClassifier()

    train_events, train_labels = train_cases
    for event, label in zip(train_events, train_labels):
        classifier.fit(event, label)

    expected_results = [0.20232035389852276, 0.1385666227211696, 0.11103582174277878]
    actual_results = [classifier.predict_proba(event) for event in test_events]

    assert expected_results == actual_results


def test_interest_classifier(train_cases, test_events):
    classifier = learning.InterestClassifier(positive_only=False)

    train_events, train_labels = train_cases
    for event, label in zip(train_events, train_labels):
        classifier.fit(event, label)

    expected_results = [0.8253463535369645, 0.8124833431982229, 0.7434285783213926]
    actual_results = [classifier.predict_proba(event) for event in test_events]

    assert expected_results == actual_results
