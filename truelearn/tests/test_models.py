# pylint: disable=missing-function-docstring,missing-class-docstring
import collections

from truelearn import models


class TestLearnerModel:
    def test_learner_model_default_construct(self):
        model = models.LearnerModel()

        assert model.number_of_engagements == 0
        assert model.number_of_non_engagements == 0
        assert isinstance(model.knowledge, models.Knowledge)


class TestLearnerMetaModel:
    def test_learner_meta_model_default_construct(self):
        model = models.LearnerMetaModel()

        assert (
            model.bias_weight
            == model.interest_weight
            == model.novelty_weight
            == models.LearnerMetaModel.Weights()
        )
        assert isinstance(model.learner_novelty, models.LearnerModel)
        assert isinstance(model.learner_interest, models.LearnerModel)


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

        assert kc.mean == 1
        assert kc.variance == 2
        assert kc.timestamp == 3
        assert kc.title == "Hello World"
        assert kc.description is None
        assert kc.url is None

    def test_knowledge_component_update(self):
        kc = models.KnowledgeComponent(mean=1.0, variance=0.5, timestamp=None)
        kc.update(mean=0.0, variance=1.0, timestamp=1)

        assert kc.mean == 0
        assert kc.variance == 1
        assert kc.timestamp == 1
        assert kc.title is None
        assert kc.description is None
        assert kc.url is None

    def test_knowledge_component_clone(self):
        kc = models.KnowledgeComponent(mean=1.0, variance=0.5)
        kc_cloned = kc.clone(mean=2.0, variance=3.0, timestamp=None)

        assert kc_cloned.mean == 2
        assert kc_cloned.variance == 3
        assert kc_cloned.timestamp is None
        assert kc_cloned.title is None
        assert kc_cloned.description is None
        assert kc_cloned.url is None

    def test_knowledge_component_export_as_dict(self):
        kc = models.KnowledgeComponent(mean=1.0, variance=0.5)
        params = kc.export_as_dict()

        assert params == {
            "mean": 1,
            "variance": 0.5,
            "timestamp": None,
            "title": None,
            "description": None,
            "url": None,
        }

    def test_knowledge_component_repr(self, capsys):
        kc = models.KnowledgeComponent(mean=1.0, variance=0.5)
        print(kc)

        captured = capsys.readouterr()
        assert (
            captured.out
            == "KnowledgeComponent(mean=1.0, variance=0.5, timestamp=None, title=None, \
description=None, url=None)\n"
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

        assert kc.mean == 1
        assert kc.variance == 2
        assert kc.timestamp == 3
        assert kc.title == "Hello World"
        assert kc.description is None
        assert kc.url is None
        assert kc.history == collections.deque()

    def test_history_knowledge_component_update(self):
        kc = models.HistoryAwareKnowledgeComponent(
            mean=1.0, variance=0.5, timestamp=None, history=collections.deque()
        )
        kc.update(mean=0.0, variance=1.0, timestamp=1)

        assert kc.mean == 0
        assert kc.variance == 1
        assert kc.timestamp == 1
        assert kc.title is None
        assert kc.description is None
        assert kc.url is None
        assert kc.history == collections.deque([(1, 0.5, None)])

    def test_history_knowledge_component_clone(self):
        kc = models.HistoryAwareKnowledgeComponent(mean=1.0, variance=0.5)
        kc_cloned = kc.clone(mean=2.0, variance=3.0, timestamp=None)

        assert kc_cloned.mean == 2
        assert kc_cloned.variance == 3
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
            "mean": 1,
            "variance": 0.5,
            "timestamp": None,
            "title": None,
            "description": None,
            "url": None,
            "history": collections.deque([(0, 0.5, None)]),
        }

    def test_history_knowledge_component_repr(self, capsys):
        kc = models.HistoryAwareKnowledgeComponent(mean=1.0, variance=0.5)
        print(kc)

        captured = capsys.readouterr()
        assert (
            captured.out
            == "HistoryAwareKnowledgeComponent(mean=1.0, variance=0.5, timestamp=None, \
title=None, description=None, url=None, history=deque([], maxlen=None))\n"
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

    def test_knowledge_repr(self, capsys):
        knowledge = models.Knowledge(
            {1: models.KnowledgeComponent(mean=0.0, variance=0.0)}
        )
        print(knowledge)

        captured = capsys.readouterr()
        assert (
            captured.out
            == "Knowledge(knowledge={1: KnowledgeComponent(mean=0.0, variance=0.0, \
timestamp=None, title=None, description=None, url=None)})\n"
        )
