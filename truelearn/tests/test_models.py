# pylint: disable=missing-function-docstring,missing-class-docstring
import collections

import pytest

from truelearn import models


class TestLearnerModel:
    def test_learner_model_default_construct(self):
        model = models.LearnerModel()

        assert model.number_of_engagements == 0
        assert model.number_of_non_engagements == 0
        assert isinstance(model.knowledge, models.Knowledge)


class TestLearnerMetaWeights:
    def test_learner_meta_model_default_construct(self):
        weights = models.LearnerMetaWeights()

        assert (
            weights.bias_weights
            == weights.interest_weights
            == weights.novelty_weights
            == models.LearnerMetaWeights.Weights()
        )


class TestEventModel:
    def test_event_model_default_construct(self):
        model = models.EventModel()

        assert model.event_time is None
        assert isinstance(model.knowledge, models.Knowledge)


class TestKnowledgeComponent:
    def test_knowledge_component_construct_with_param(self):
        kc = models.KnowledgeComponent(
            mean=1.0,
            variance=2.0,
            timestamp=3.0,
            title="Hello World",
            description=None,
            url=None,
        )

        assert kc.mean == 1.0
        assert kc.variance == 2.0
        assert kc.timestamp == 3.0
        assert kc.title == "Hello World"
        assert kc.description is None
        assert kc.url is None

    def test_knowledge_component_update(self):
        kc = models.KnowledgeComponent(mean=1.0, variance=0.5, timestamp=None)
        kc.update(variance=1.0, timestamp=1)

        assert kc.mean == 1.0
        assert kc.variance == 1.0
        assert kc.timestamp == 1.0
        assert kc.title is None
        assert kc.description is None
        assert kc.url is None

        kc.update(mean=0.0)
        assert kc.mean == 0.0

    def test_knowledge_component_clone(self):
        kc = models.KnowledgeComponent(mean=1.0, variance=0.5)
        kc_cloned = kc.clone(mean=2.0, variance=3.0, timestamp=None)

        assert kc_cloned.mean == 2.0
        assert kc_cloned.variance == 3.0
        assert kc_cloned.timestamp is None
        assert kc_cloned.title is None
        assert kc_cloned.description is None
        assert kc_cloned.url is None

    def test_knowledge_component_export_as_dict(self):
        kc = models.KnowledgeComponent(mean=1.0, variance=0.5)
        params = kc.export_as_dict()

        assert params == {
            "mean": 1.0,
            "variance": 0.5,
            "timestamp": None,
            "title": None,
            "description": None,
            "url": None,
        }

    def test_knowledge_component_repr(self):
        kc = models.KnowledgeComponent(mean=1.0, variance=0.5)
        out = repr(kc)

        assert (
            out
            == "KnowledgeComponent(mean=1.0, variance=0.5, timestamp=None, title=None, \
description=None, url=None)"
        )


class TestHistoryKnowledgeComponent:
    def test_history_knowledge_component_construct_with_param(self):
        kc = models.HistoryAwareKnowledgeComponent(
            mean=1.0,
            variance=2.0,
            timestamp=3.0,
            title="Hello World",
            description=None,
            url=None,
        )

        assert kc.mean == 1.0
        assert kc.variance == 2.0
        assert kc.timestamp == 3.0
        assert kc.title == "Hello World"
        assert kc.description is None
        assert kc.url is None
        assert kc.history == collections.deque()

    def test_history_knowledge_component_construct_with_deque(self):
        kc = models.HistoryAwareKnowledgeComponent(
            mean=1.0,
            variance=2.0,
            history=collections.deque(
                [(float(i), float(i), float(i)) for i in range(101)]
            ),
            history_limit=10,
        )

        assert kc.mean == 1.0
        assert kc.variance == 2.0
        assert kc.history.maxlen == 10
        assert list(kc.history) == [
            (float(i), float(i), float(i)) for i in range(91, 101)
        ]

    def test_history_knowledge_component_update(self):
        kc = models.HistoryAwareKnowledgeComponent(
            mean=1.0, variance=0.5, timestamp=None, history=collections.deque()
        )
        kc.update(mean=0.0, variance=1.0, timestamp=1)

        assert kc.mean == 0.0
        assert kc.variance == 1.0
        assert kc.timestamp == 1.0
        assert kc.title is None
        assert kc.description is None
        assert kc.url is None
        assert kc.history == collections.deque([(1, 0.5, None)])

    def test_history_knowledge_component_clone(self):
        kc = models.HistoryAwareKnowledgeComponent(mean=1.0, variance=0.5)
        kc_cloned = kc.clone(mean=2.0, variance=3.0, timestamp=None)

        assert kc_cloned.mean == 2.0
        assert kc_cloned.variance == 3.0
        assert kc_cloned.timestamp is None
        assert kc_cloned.title is None
        assert kc_cloned.description is None
        assert kc_cloned.url is None
        assert kc_cloned.history == collections.deque()

    def test_history_knowledge_component_export_as_dict(self):
        kc = models.HistoryAwareKnowledgeComponent(
            mean=1.0, variance=0.5, history=collections.deque([(0, 0.5, None)])
        )
        params = kc.export_as_dict()

        assert params == {
            "mean": 1.0,
            "variance": 0.5,
            "timestamp": None,
            "title": None,
            "description": None,
            "url": None,
            "history": collections.deque([(0, 0.5, None)]),
        }

    def test_history_knowledge_component_repr(self):
        kc = models.HistoryAwareKnowledgeComponent(mean=1.0, variance=0.5)
        out = repr(kc)

        assert (
            out
            == "HistoryAwareKnowledgeComponent(mean=1.0, variance=0.5, timestamp=None, \
title=None, description=None, url=None, history=deque([], maxlen=None))"
        )

    def test_history_knowledge_component_repr_with_max_object(self):
        kc = models.HistoryAwareKnowledgeComponent(
            mean=1.0,
            variance=2.0,
            history=collections.deque(
                [(float(i), float(i), float(i)) for i in range(101)]
            ),
            history_limit=10,
        )

        out = kc.__repr__(2)  # pylint: disable=unnecessary-dunder-call

        assert (
            out
            == "HistoryAwareKnowledgeComponent(mean=1.0, variance=2.0, timestamp=None, \
title=None, description=None, url=None, history=\
deque([(91.0, 91.0, 91.0), (92.0, 92.0, 92.0), ...], maxlen=10))"
        )

        with pytest.raises(ValueError) as excinfo:
            kc.__repr__(-1)  # pylint: disable=unnecessary-dunder-call
        assert "Expected n_max_object>=0. Got n_max_object=-1 instead." == str(
            excinfo.value
        )


class TestKnowledge:
    def test_knowledge_construction(self):
        models.Knowledge()
        models.Knowledge({})

    def test_knowledge_get_kc(self):
        knowledge = models.Knowledge()
        default_kc = models.KnowledgeComponent(mean=0.5, variance=1)
        assert knowledge.get_kc(1, default_kc) == default_kc

    def test_knowledge_knowledge_components(self):
        knowledge = models.Knowledge()
        assert not list(knowledge.knowledge_components())

        kc = models.KnowledgeComponent(mean=0.0, variance=0.5)
        knowledge = models.Knowledge({1: kc})
        assert len(list(knowledge.knowledge_components())) == 1
        assert list(knowledge.knowledge_components())[0] == kc

    def test_knowledge_topic_kc_pairs(self):
        knowledge = models.Knowledge()
        assert not list(knowledge.topic_kc_pairs())

        kc = models.KnowledgeComponent(mean=0.0, variance=0.5)
        knowledge = models.Knowledge({1: kc})
        assert len(list(knowledge.topic_kc_pairs())) == 1
        assert list(knowledge.topic_kc_pairs())[0] == (1, kc)

    def test_knowledge_update_kcs(self):
        knowledge = models.Knowledge(
            {1: models.KnowledgeComponent(mean=0.0, variance=0.0)}
        )

        kc = models.KnowledgeComponent(mean=1.0, variance=0.5)
        knowledge.update_kc(1, kc)
        assert (
            knowledge.get_kc(1, models.KnowledgeComponent(mean=0.0, variance=0.5)) == kc
        )

    def test_knowledge_repr(self):
        knowledge = models.Knowledge(
            {1: models.KnowledgeComponent(mean=0.0, variance=0.0)}
        )
        out = repr(knowledge)

        assert (
            out
            == "Knowledge(knowledge={1: KnowledgeComponent(mean=0.0, variance=0.0, \
timestamp=None, title=None, description=None, url=None)})"
        )

    def test_knowledge_repr_with_max_objects(self):
        knowledge = models.Knowledge(
            {
                1: models.KnowledgeComponent(mean=0.0, variance=0.0),
                2: models.KnowledgeComponent(mean=1.0, variance=1.0),
            }
        )
        out = knowledge.__repr__(2)  # pylint: disable=unnecessary-dunder-call

        assert (
            out
            == "Knowledge(knowledge={1: KnowledgeComponent(mean=0.0, variance=0.0, \
timestamp=None, title=None, description=None, url=None), 2: KnowledgeComponent(\
mean=1.0, variance=1.0, timestamp=None, title=None, description=None, url=None)})"
        )

        with pytest.raises(ValueError) as excinfo:
            knowledge.__repr__(-1)  # pylint: disable=unnecessary-dunder-call
        assert "Expected n_max_object>=0. Got n_max_object=-1 instead." == str(
            excinfo.value
        )
