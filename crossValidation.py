import pandas as pd
from DecisionTree import *


def get_class_counts(df, target_attr):
    class_counts = {}
    for value in df[target_attr]:
        if value not in class_counts:
            class_counts[value] = 0
        class_counts[value] += 1
    return class_counts


def classify(instance, tree, data):
    # Traverse the tree until a leaf node is reached
    target_attr = "class"
    while not tree.is_leaf:
        attribute_value = instance[tree.attribute]
        if attribute_value not in tree.children:
            # If the attribute value is not present in the tree,
            # return the most common class label of the training data
            return max(
                get_class_counts(data, target_attr),
                key=get_class_counts(data, target_attr).get,
            )
        tree = tree.children[attribute_value]
    return tree.label


def error_rate(tree, df):
    # classify each example in the dataframe and compare with actual labels
    predicted_labels = df.apply(lambda x: classify(x, tree, df), axis=1)
    actual_labels = df["class"]
    misclassified = (predicted_labels != actual_labels).sum()
    return misclassified / len(df)


def cross_validation(examples, k):
    n = len(examples)
    errs = 0
    for i in range(k):
        # split data into training and validation sets
        start = int((i * n) / k)
        end = int(((i + 1) * n) / k)
        validation_set = examples.iloc[start:end]
        training_set = pd.concat([examples.iloc[:start], examples.iloc[end:]])

        # train a decision tree using the learner function
        learner = DecisionTree(training_set, list(training_set)).tree

        # compute error rate on validation set using the error_rate function
        errs += error_rate(learner, validation_set)

    return errs / k
