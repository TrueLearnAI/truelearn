# pylint: disable=missing-function-docstring
# noqa
"""
TrueLearn Experiments
=====================

This example replicates a simple version of truelearn experiments,
measuring the accuracy, precision, recall and f1 score of
each classifier inside the truelearn package.

The data used in the example is the PEEK Dataset, imported via
``truelearn.datasets`` subpackage.
It is worth noting that because of the large amount of data in the dataset,
this example uses multiple processes for training.

The two tables below show the weighted average test performance for
accuracy, precision, recall and F1.

.. list-table:: All learners
    :header-rows: 1

    * - Classifier
      - Accuracy
      - Precision
      - Recall
      - F1 Score
    * - EngageClassifier
      - 0.574
      - 0.574
      - 0.898
      - 0.663
    * - PersistentClassifier
      - 0.779
      - 0.638
      - 0.637
      - 0.636
    * - MajorityClassifier
      - 0.743
      - 0.517
      - 0.561
      - 0.525
    * - KnowledgeClassifier
      - 0.671
      - 0.590
      - 0.551
      - 0.545
    * - NoveltyClassifier
      - 0.573
      - 0.575
      - 0.769
      - 0.620
    * - InterestClassifier
      - 0.569
      - 0.577
      - 0.681
      - 0.583
    * - INKClassifier
      - 0.761
      - 0.559
      - 0.603
      - 0.568

.. list-table:: 20 most active learners
    :header-rows: 1

    * - Classifier
      - Accuracy
      - Precision
      - Recall
      - F1 Score
    * - EngageClassifier
      - 0.637
      - 0.637
      - 1.000
      - 0.737
    * - PersistentClassifier
      - 0.750
      - 0.724
      - 0.723
      - 0.723
    * - MajorityClassifier
      - 0.766
      - 0.618
      - 0.719
      - 0.648
    * - KnowledgeClassifier
      - 0.684
      - 0.652
      - 0.870
      - 0.721
    * - NoveltyClassifier
      - 0.654
      - 0.645
      - 0.900
      - 0.720
    * - InterestClassifier
      - 0.634
      - 0.640
      - 0.930
      - 0.719
    * - INKClassifier
      - 0.795
      - 0.697
      - 0.748
      - 0.696

It can be noted that the results presented in the table are slightly different
from the data in the TrueLearn paper because here we did not use train datasets
for hyperparameter training. If hyperparameter training is done, we expect that
the final parameters will be very close to the data in the paper.
"""

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
    # train is for hyperparameter tuning
    # test is for evaluation
    #
    # we neglect the tuning process and use the default values here
    _, test, _ = datasets.load_peek_dataset(train_limit=0)

    all_data = test
    def_var = get_dataset_variance(all_data)

    # you can select top n active users (active => more activities)
    # by default 20
    top_n = 20
    all_data.sort(key=lambda t: len(t[1]), reverse=True)
    all_data = all_data[:top_n]

    # the value 1 and 400 below is randomly chosen
    # the appropriate value can be found by using hyperparameter training
    classifier_cls_list = [
        learning.EngageClassifier(),
        learning.PersistentClassifier(),
        learning.MajorityClassifier(),
        learning.KnowledgeClassifier(def_var=def_var * 400),
        learning.NoveltyClassifier(def_var=def_var * 1),
        learning.InterestClassifier(def_var=def_var * 400),
        learning.INKClassifier(
            novelty_classifier=learning.NoveltyClassifier(def_var=def_var * 1),
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
