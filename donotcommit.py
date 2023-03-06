from truelearn.datasets import load_peek_dataset
from truelearn.learning import (
    KnowledgeClassifier,
    # NoveltyClassifier
) 
from truelearn.models import (
    # Knowledge,
    HistoryAwareKnowledgeComponent,
)
from truelearn.utils.visualisations import (
    knowledge_to_dict,
    BarPlotter,
    LinePlotter,
    DotPlotter
)
 
 
def get_data():
    data = load_peek_dataset(kc_init_func=get_history_kc)
    return data
 
 
def train_model(model, data):
    """
    x = the EventModel object
    y = bool of whether the user has engaged
    """
    for x, y  in data:
        model.fit(x, y)
 
 
def get_history_kc(mean, variance, timestamp, url, title, description):
    return HistoryAwareKnowledgeComponent(
        mean=mean, 
        variance=variance, 
        timestamp=timestamp, 
        url=url,
        title=title,
        description=description
    )
 
 
if __name__ == '__main__':
    data = get_data()
    train_values = data[0]
    test_values = data[1]
    id_map = data[2]
    learner_events = train_values[12]
    model = KnowledgeClassifier()
    train_model(model, learner_events[1])
    learner = model.get_learner_model()
    kd = knowledge_to_dict(learner.knowledge)
    bp = DotPlotter()
    # bp = BarPlotter()
    data = bp.clean_data(kd, 5)
    #print(data)
    layout = ("Comparison of learner's top 5 subjects", "Subjects", "Mean") #edit title
    bp.plot(layout, data).show()