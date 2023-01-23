class LearnerDataModel:

    def __init__(self, learner_id, learner_data, skill=0., variance=0., tau=0., beta_squared=0.):
        self._learner_id = learner_id
        self._learner_data = learner_data
        self._skill = skill
        self._variance = variance
        self._tau = tau
        self._beta_squared = beta_squared

    def get_learner_id(self):
        return self._learner_id

    def get_learner_data(self):
        return self._learner_data

    def update_skill(self, skill):
        self._skill = skill
