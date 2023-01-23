from truelearn.learner_models.learner_data import LearnerDataModel
from truelearn.metrics.confusion_matrix import ConfusionMatrix


class TrueLearnNovelClassifier:

    # Default create a new model
    def __init__(self, learner_model=LearnerDataModel()) -> None:
        self._learner_model = learner_model

    def fit(self) -> None:
        pass

    # Binary classification, yes or no (true or false)
    def predict(self) -> bool:
        pass

    def predict_proba(self) -> float:
        pass

    def score(self) -> ConfusionMatrix:
        pass
