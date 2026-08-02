import numpy as np


def calculate_accuracy(y_true, y_pred):
    correct = 0
    total = len(y_true)
    for i in range(total):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / total


def calculate_precision(y_true, y_pred):
    true_positive = 0
    false_positive = 0
    for i in range(len(y_true)):
        if y_pred[i] == 1:
            if y_true[i] == 1:
                true_positive += 1
            else:
                false_positive += 1
    return true_positive / (true_positive + false_positive)


def calculate_recall(y_true, y_pred):
    true_positive = 0
    false_negative = 0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] == 1:
                true_positive += 1
            else:
                false_negative += 1
    return true_positive / (true_positive + false_negative)


def calculate_f1_score(y_true, y_pred):
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)
