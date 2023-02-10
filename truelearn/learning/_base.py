from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, Hashable
from statistics import NormalDist
from math import sqrt
from collections import defaultdict

import trueskill

from truelearn.models import EventModel, AbstractKnowledge, AbstractKnowledgeComponent, LearnerModel


# pylint: disable=pointless-string-statement
'''
BSD 3-Clause License

Copyright (c) 2007-2022 The scikit-learn developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Copyright for `get_params` and `set_params` method of the BaseClassifier
are held by [BSD 3-Clause License, scikit-learn developers, 2007-2022].
The other parts of this file are licensed under MIT License (LICENSE).
'''


class BaseClassifier(ABC):
    """The base class of all the classifiers in TrueLearn.

    Every subclass of the BaseClassifier should define their
    `_parameter_constraints`. This specified the parameters exposed
    via the `get_params` method and the constraints on the type of
    the parameters.

    The `_parameter_constraints` is a dictionary that maps parameter
    names to its expected type. The expected type can be a list or a single type
    as it's possible for a type to accept more than one type.
    To do the constraint check based on this, simply call `self._validate_params` in your classifier.

    If you don't want to do the constraint check, but want to support `get_params`,
    you don't need to call the `self._validate_params()`.

    """

    __DEEP_PARAM_DELIMITER: str = "__"
    _parameter_constraints: dict = {}

    @abstractmethod
    def fit(self, x: EventModel, y: bool) -> BaseClassifier:
        """Train the model based on a given EventModel that represents a learning event.

        Parameters
        ----------
        x : EventModel
            A representation of a learning event.
        y : bool
            Whether the learner engages in the given learning event.

        Returns
        -------
        BaseClassifier
            The updated BaseClassifier.

        """

    @abstractmethod
    def predict(self, x: EventModel) -> bool:
        """Predict whether the learner will engage in the given learning event.

        Parameters
        ----------
        x : EventModel
            A representation of a learning event.

        Returns
        -------
        bool
            Whether the learner will engage in the given learning event.

        """

    @abstractmethod
    def predict_proba(self, x: EventModel) -> float:
        """Predict the probability of the learner's engagement in the given learning event.

        Parameters
        ----------
        x : EventModel
            A representation of a learning event.

        Returns
        -------
        float
            The probability that the learner engages in the given learning event.

        """

    def get_params(self, deep=True):
        """Get parameters for Classifier.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this Classifier and
            contained sub-objects that inherits BaseClassifier class.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.

        """
        param_names = list(self._parameter_constraints.keys())
        param_names.sort()

        out = {}
        for key in param_names:
            if not hasattr(self, key):
                raise ValueError(
                    f"The specified parameter name {key} is not in the {self.__class__.__name__}")

            value = getattr(self, key)
            if deep and isinstance(value, BaseClassifier):
                deep_items = value.get_params().items()
                out.update((key + BaseClassifier.__DEEP_PARAM_DELIMITER + k, val)
                           for k, val in deep_items)
            out[key] = value

        return out

    def set_params(self, **params):
        """Set the parameters of the Classifier.

        Parameters
        ----------
        **params : dict[str, Any]
            The parameters for Classifier.

        Returns
        -------
        self : Self
            The updated Classifier.

        """
        # avoid running `self.get_params` if there is no given params
        if not params:
            return self

        valid_params = self.get_params(deep=True)

        # a dictionary that stores params for nested classifiers
        # it stores a map from nested_classifier_name to its parameters (a dict)
        # { nested_classifier_name => {key => value} }
        nested_params = defaultdict(dict)

        for key, value in params.items():
            key, delim, sub_key = key.partition(
                BaseClassifier.__DEEP_PARAM_DELIMITER)
            if key not in valid_params:
                raise ValueError(
                    f"The given parameter {key} is not in the class {self.__class__.__name__}. "
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def _validate_params(self):
        """Validate types of constructor parameters."""
        params = self.get_params(deep=False)

        for param_name, expected_param_type in self._parameter_constraints.items():
            # ensure param_name is in the valid params dictionary
            if param_name not in params:
                raise ValueError(
                    f"The {param_name} parameter is not in the class {self.__class__.__name__}."
                )

            # ensure expected_param_type is properly set
            # `isinstance(expected_param_type, type)` works for python 3
            if isinstance(expected_param_type, list):
                # check if all the element inside the list are classes
                if not all(isinstance(param_type_unpacked, type) for param_type_unpacked in expected_param_type):
                    raise TypeError(
                        "The given constraint list contains non-class element."
                    )
            else:
                # check if expected_param_type is a class
                if not isinstance(expected_param_type, type):
                    raise ValueError(
                        f"The given constraint {expected_param_type} is not a class."
                    )

            param = getattr(self, param_name)

            if isinstance(expected_param_type, list):
                # if match non of the type in the constraints
                if not any(isinstance(param, param_type_unpacked) for param_type_unpacked in list(expected_param_type)):
                    param_classname_expected = list(
                        map(lambda cls: cls.__name__, expected_param_type))
                    raise TypeError(
                        f"The {param_name} parameter of {self.__class__.__name__} must be"
                        f" one of the classes in {param_classname_expected}."
                        f" Got {param.__class__.__name__} instead."
                    )
            else:
                if not isinstance(param, expected_param_type):
                    raise TypeError(
                        f"The {param_name} parameter of {self.__class__.__name__} must be"
                        f" {expected_param_type.__name__}. Got {param.__class__.__name__} instead."
                    )


class InterestNoveltyKnowledgeBaseClassifier(BaseClassifier):
    """A Base Classifier for KnowledgeClassifier, NoveltyClassifier and InterestClassifier.

    It defines the necessary instance variables and
    common methods to interact with the LearnerModel.

    Parameters
    ----------
    learner_model: LearnerModel | None, optional
    threshold: float
        Threshold for judging learner engagement. If the probability of the learner engagement is greater
        than the threshold, the model will predict engagement.
    init_skill: float
        The initial skill (mean) of the learner given a new KnowledgeComponent.
    def_var: float
        The default variance of the new KnowledgeComponent.
    beta: float
        The noise factor, which is used in trueskill.
    positive_only: bool
        Whether the model updates itself only if encountering positive data.

    # TODO: fix probability doc

    Methods
    -------
    fit(x, y)
        Train the model based on the given data and label.
    predict(x)
        Predict whether the learner will engage.
    predict_proba(x)
        Predict the probability of learner engagement.
    get_params()
        Get parameters associated with the model.

    """

    DEFAULT_CONTENT_VARIANCE: float = 1e-9
    DEFAULT_DRAW_PROBA_LOW: float = 1e-9
    DEFAULT_DRAW_PROBA_HIGH: float = 0.999999999

    _parameter_constraints: dict = {
        **BaseClassifier._parameter_constraints,
        "_learner_model": [LearnerModel, type(None)],
        "_threshold": float,
        "_init_skill": float,
        "_def_var": float,
        "_beta": float,
        "_positive_only": bool,
        "_draw_proba_type": str,
        "_draw_proba_static": [float, type(None)],
        "_draw_proba_factor": float
    }

    def __init__(self, *, learner_model: LearnerModel | None, threshold: float,
                 init_skill: float, def_var: float, beta: float, positive_only: bool,
                 draw_proba_type: str, draw_proba_static: float | None, draw_proba_factor: float) -> None:
        super().__init__()

        if learner_model is None:
            self._learner_model = LearnerModel()
        else:
            self._learner_model = learner_model
        self._threshold = threshold
        self._init_skill = init_skill
        self._def_var = def_var
        self._beta = beta
        self._positive_only = positive_only

        self._draw_proba_type = draw_proba_type
        self._draw_proba_factor = draw_proba_factor

        if self._draw_proba_type not in ("static", "dynamic"):
            raise ValueError(
                "Unsupported draw probability type. The model supports two types: static and dynamic")

        self._draw_proba_static = draw_proba_static
        self._draw_probability = self.__calculate_draw_proba()

        # create an environment in which training will take place
        self._env = trueskill.TrueSkill()

    def __calculate_draw_proba(self):
        if self._draw_proba_type == "static":
            # delayed check as this can be potentially replaced by set_params
            if self._draw_proba_static is None:
                raise ValueError(
                    "When _draw_proba_type is set to static, the draw_proba_static should not be None"
                )
            return self._draw_proba_static * self._draw_proba_factor

        total_engagement_stats = max(
            1, self._learner_model.number_of_engagements + self._learner_model.number_of_non_engagements)
        draw_probability = float(
            self._learner_model.number_of_engagements / total_engagement_stats)

        # clamp the value between [DEFAULT_DRAW_PROBA_LOW, DEFAULT_DRAW_PROBA_HIGH]
        draw_probability = max(min(InterestNoveltyKnowledgeBaseClassifier.DEFAULT_DRAW_PROBA_HIGH,
                               draw_probability), InterestNoveltyKnowledgeBaseClassifier.DEFAULT_DRAW_PROBA_LOW)

        # draw_proba_param is a factor if the type is dynamic
        return draw_probability * self._draw_proba_factor

    def __setup_env(self) -> None:
        self.__calculate_draw_proba()
        trueskill.setup(mu=0., sigma=InterestNoveltyKnowledgeBaseClassifier.DEFAULT_CONTENT_VARIANCE, beta=float(self._beta),
                        tau=float(self._learner_model.tau), draw_probability=self._draw_probability,
                        backend="mpmath", env=self._env)

    def __update_engagement_stats(self, y) -> None:
        if y:
            self._learner_model.number_of_engagements += 1
        else:
            self._learner_model.number_of_non_engagements += 1

    def _gather_trueskill_team(self, kcs: Iterable[AbstractKnowledgeComponent]) -> tuple[trueskill.Rating]:
        return tuple(map(
            lambda kc: self._env.create_rating(
                mu=kc.mean, sigma=sqrt(kc.variance)),
            kcs
        ))

    @abstractmethod
    def _update_knowledge_representation(self, x, y) -> None:
        """Update the knowledge representation of the LearnerModel.

        Parameters
        ----------
        x : EventModel
            A representation of a learning event.
        y : bool
            Whether the learner engages in the given learning event.

        """

    def fit(self, x: EventModel, y: bool) -> InterestNoveltyKnowledgeBaseClassifier:
        """Train the model based on a given EventModel that represents a learning event.

        Parameters
        ----------
        x : EventModel
            A representation of a learning event.
        y : bool
            Whether the learner engages in the given learning event.

        Returns
        -------
        InterestNoveltyKnowledgeBaseClassifier
            The updated InterestNoveltyKnowledgeBaseClassifier.

        """
        # update the knowledge representation if positive_only is False or (it's true and y is true)
        if not self._positive_only or y is True:
            self.__setup_env()
            self._update_knowledge_representation(x, y)

        self.__update_engagement_stats(y)
        return self

    def predict(self, x: EventModel) -> bool:
        """Predict whether the learner will engage in the given learning event.

        The function will return True iff the probability that the learner engages
        with the learnable unit is greater than the given threshold.

        Refer to `predict_proba` for more details.

        Parameters
        ----------
        x : EventModel
            A representation of a learning event.

        Returns
        -------
        bool
            Whether the learner will engage in the given learning event.

        """
        return self.predict_proba(x) > self._threshold


def team_sum_quality(learner_kcs: Iterable[AbstractKnowledgeComponent], content_kcs: Iterable[AbstractKnowledgeComponent], beta: float) -> float:
    """Return the probability that the learner engages with the learnable unit.

    Parameters
    ----------
    learner_kcs : Iterable[AbstractKnowledgeComponent]
        An iterable of learner's knowledge component.
    content_kcs : Iterable[AbstractKnowledgeComponent]
        An iterable of learnable unit's knowledge component.
    beta : float
        The noise factor, which is used in trueskill.

    Returns
    -------
    float

    """
    team_learner_mean = map(lambda kc: kc.mean, learner_kcs)
    team_learner_variance = map(lambda kc: kc.variance, learner_kcs)
    team_content_mean = map(lambda kc: kc.mean, content_kcs)
    team_content_variance = map(lambda kc: kc.variance, content_kcs)

    difference = sum(team_learner_mean) - sum(team_content_mean)
    std = sqrt(sum(team_learner_variance) + sum(team_content_variance) + beta)
    return NormalDist(mu=0, sigma=std).cdf(difference)


def select_topic_kc_pairs(learner_model: LearnerModel, content_knowledge: AbstractKnowledge,
                          init_skill: float, def_var: float, def_timestamp: float | None = None) -> Iterable[tuple[Hashable, AbstractKnowledgeComponent]]:
    """Return an iterable representing the learner's knowledge in the topics specified by the learnable unit.

    Given the knowledge representation of the learnable unit, this method tries to get
    the corresponding knowledge representation from the Learner Model.

    If it cannot find the corresponding knowledge component in learner's model, which means
    the learner has never exposed to this knowledge component before, a new KC will be constructed
    with initial skill and default variance. This will be done by using `__topic_kc_pair_mapper`.

    Parameters
    ----------
    learner_model : LearnerModel
        A representation of the learner.
    content_knowledge : AbstractKnowledge
        The knowledge representation of a learnable unit.
    init_skill: float
        The initial skill (mean) of the learner given a new AbstractKnowledgeComponent.
    def_var: float
        The default variance of the new AbstractKnowledgeComponent.
    def_timestamp: float | None, optional
        The default timestamp of the new AbstractKnowledgeComponent.
        It should be the event time for InterestClassifier.
        This field could be left empty if InterestClassifier or any meta model
        that uses InterestClassifier are not used.

    Returns
    -------
    Iterable[tuple[Hashable, AbstractKnowledgeComponent]]

    """
    def __topic_kc_pair_mapper(topic_kc_pair: tuple[Hashable, AbstractKnowledgeComponent]) -> tuple[Hashable, AbstractKnowledgeComponent]:
        topic_id, kc = topic_kc_pair
        extracted_kc = learner_model.knowledge.get_kc(
            topic_id, kc.clone(mean=init_skill, variance=def_var, timestamp=def_timestamp))
        return topic_id, extracted_kc

    team_learner = map(__topic_kc_pair_mapper,
                       content_knowledge.topic_kc_pairs())
    return team_learner


def select_kcs(learner_model: LearnerModel, content_knowledge: AbstractKnowledge,
               init_skill: float, def_var: float, def_timestamp: float | None = None) -> Iterable[AbstractKnowledgeComponent]:
    """Return an iterable representing the learner's knowledge in the topics specified by the learnable unit.

    Given the knowledge representation of the learnable unit, this method tries to get
    the corresponding knowledge representation from the Learner Model.

    If it cannot find the corresponding knowledge component in learner's model, which means
    the learner has never exposed to this knowledge component before, a new KC will be constructed
    with initial skill and default variance. This will be done by using `__topic_kc_pair_mapper`.

    Parameters
    ----------
    learner_model : LearnerModel
        A representation of the learner.
    content_knowledge : AbstractKnowledge
        The knowledge representation of a learnable unit.
    init_skill: float
        The initial skill (mean) of the learner given a new AbstractKnowledgeComponent.
    def_var: float
        The default variance of the new AbstractKnowledgeComponent.
    def_timestamp: float | None, optional
        The default timestamp of the new AbstractKnowledgeComponent.
        It should be the event time for InterestClassifier.
        This field could be left empty if InterestClassifier or any meta model
        that uses InterestClassifier are not used.

    Returns
    -------
    Iterable[tuple[Hashable, AbstractKnowledgeComponent]]

    """
    def __kc_mapper(topic_kc_pair: tuple[Hashable, AbstractKnowledgeComponent]) -> AbstractKnowledgeComponent:
        topic_id, kc = topic_kc_pair
        extracted_kc = learner_model.knowledge.get_kc(
            topic_id, kc.clone(mean=init_skill, variance=def_var, timestamp=def_timestamp))
        return extracted_kc

    team_learner = map(__kc_mapper, content_knowledge.topic_kc_pairs())
    return team_learner
