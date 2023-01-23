from truelearn.learner_models.learner_data import LearnerDataModel


class TrueLearnKnowledgeClassifier:

    # Default create a new model
    def __init__(self, learner_model=LearnerDataModel()):
        self._learner_model = learner_model

    def fit(self):
        pass

    def predict(self):
        pass

    def predict_proba(self):
        pass

    def score(self):
        pass
