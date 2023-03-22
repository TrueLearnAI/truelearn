# pylint: disable=missing-function-docstring
import copy
import itertools
import multiprocessing
from typing import List, Tuple

from truelearn import learning, datasets, preprocessing
from truelearn.utils import metrics


def get_dataset_variance(dataset: datasets.PEEKData):
    coverages = []
    for _, events in dataset:
        events = map(lambda t: t[0], events)
        knowledges = map(lambda e: e.knowledge.knowledge_components(), events)

        for kcs in knowledges:
            coverages += map(lambda kc: kc.mean, kcs)
    return preprocessing.get_values_population_std(coverages) ** 2


def fit_predict_and_transform(
    dataset: datasets.PEEKData,
    classifier_temp: learning.BaseClassifier,
) -> List[Tuple[List[bool], List[bool]]]:
    results_for_each_learner = []
    for _, event_label_pairs in dataset:
        # use the given classifier as a template to clone the new classifier
        classifier = copy.deepcopy(classifier_temp)
        actuals = []
        predictions = []

        # record if the training has already happened
        is_trained = False
        for event, label in event_label_pairs:
            # it's worth noting that there are events with empty knowledge
            # so, we need to skip these events
            if not list(event.knowledge.knowledge_components()):
                continue
            if is_trained:
                actuals.append(label)
                # prediction is based on all the previous learning events
                # see https://arxiv.org/abs/2109.03154 for more details
                predictions.append(classifier.predict(event))
            classifier.fit(event, label)
            is_trained = True

        # only append labels if they are not empty
        if actuals and predictions:
            results_for_each_learner.append((actuals, predictions))

    return results_for_each_learner


def get_classifier_metrics(
    results_for_each_learner: List[Tuple[List[bool], List[bool]]]
) -> Tuple[float, float, float, float]:
    total_count = 0

    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f1_score = 0.0

    for actuals, predictions in results_for_each_learner:
        event_count = len(predictions)
        total_count += event_count

        accuracy += event_count * metrics.get_accuracy_score(actuals, predictions)

        # we treat these values as zero if zero division happens
        precision += event_count * metrics.get_precision_score(actuals, predictions, 0)
        recall += event_count * metrics.get_recall_score(actuals, predictions, 0)
        f1_score += event_count * metrics.get_f1_score(actuals, predictions, 0)

    if total_count == 0:
        raise ValueError("No event is given")

    # weighted average
    return (
        accuracy / total_count,
        precision / total_count,
        recall / total_count,
        f1_score / total_count,
    )


def print_all_metrics(metrics: List[Tuple[str, Tuple[float, float, float, float]]]):
    headers = ["Classifier", "Accuracy", "Precision", "Recall", "F1 Score"]

    format_header = "{:>20}" * (len(headers))
    print(format_header.format(*headers))

    format_row = "{:>20}" + "{:>20.3f}" * (len(headers) - 1)
    for cls_name, metrics_for_cls in metrics:
        print(format_row.format(cls_name, *metrics_for_cls))


def multiprocessing_driver(
    all_data_classifier_pair: Tuple[datasets.PEEKData, learning.BaseClassifier]
) -> Tuple[str, Tuple[float, float, float, float]]:
    all_data, classifier = all_data_classifier_pair
    results_for_each_learner = fit_predict_and_transform(all_data, classifier)
    return classifier.__class__.__name__, get_classifier_metrics(
        results_for_each_learner
    )


def main():
    # this example is used to demonstrate the performance of our classifier
    # by using the default parameters + some handpicked values

    # train is for hyperparameter tuning
    # test is for evaluation
    #
    # we neglect the tuning process and use the default values here
    _, test, _ = datasets.load_peek_dataset(train_limit=0)

    all_data = test
    def_var = get_dataset_variance(all_data)

    # the value 1 and 400 below is randomly chosen
    # it can be found by using hyperparameter tuning
    classifier_cls_list = [
        learning.EngageClassifier(),
        learning.PersistentClassifier(),
        learning.MajorityClassifier(),
        learning.KnowledgeClassifier(def_var=def_var * 400),
        learning.NoveltyClassifier(def_var=def_var * 1),
        learning.InterestClassifier(def_var=def_var * 400),
        learning.INKClassifier(
            novelty_classifier=learning.NoveltyClassifier(def_var=def_var),
            interest_classifier=learning.InterestClassifier(def_var=def_var * 400),
        ),
    ]

    metrics_for_all = []
    # adjust this to a smaller value if you don't want to run this
    # in all your cores
    cpu_count = min(len(classifier_cls_list), multiprocessing.cpu_count())
    with multiprocessing.Pool(cpu_count) as p:
        classifier_name_metrics_pairs = p.map(
            multiprocessing_driver,
            zip(itertools.repeat(all_data), classifier_cls_list),
        )
        for cls_name, metrics_for_cls in classifier_name_metrics_pairs:
            metrics_for_all.append((cls_name, metrics_for_cls))

    print_all_metrics(metrics_for_all)


if __name__ == "__main__":
    main()
