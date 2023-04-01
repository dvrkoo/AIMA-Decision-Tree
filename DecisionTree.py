import pandas as pd
import math
import numpy as np

target_attr = "class"


def continue_to_categoric(examples, attributes):
    print("DATASET CONTAINS CONTINUOUS VALUES, APPLYING QUANTILES TO CATEGORIZE IT")
    for attribute in attributes:
        if examples[attribute].dtype == "float64":
            examples[attribute] = pd.qcut(
                examples[attribute], 6, labels=False, duplicates="drop"
            )
    print("CATEGORIZED THE DATASET")
    return examples


def load_data(file_path, is_continuous):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    # Split the DataFrame into examples and targets
    examples = df.iloc[:, :-1]
    targets = df.iloc[:, -1]
    if is_continuous:
        examples = continue_to_categoric(examples, examples.columns)
        df.iloc[:, :-1] = continue_to_categoric(
            df.iloc[:, :-1], df.iloc[:, :-1].columns
        )
    # Return the examples and targets as numpy arrays
    return examples.to_numpy(), targets.to_numpy(), df


class Node:
    def __init__(self, attribute=None, label=None, is_binary=False):
        self.attribute = attribute  # attribute used for splitting
        self.label = label  # label for leaf nodes
        self.children = {}  # dictionary of child nodes
        self.is_leaf = False  # flag to indicate if the node is a leaf
        self.type = "mul"

    def add_child(self, value, node):
        self.children[value] = node


class DecisionTree:
    def __init__(self, examples, attributes):
        self.tree = self.learn_decision_tree(examples, attributes, examples)

    def plurality_value(self, examples):
        label = examples["class"].value_counts().idxmax()
        leaf = Node(label=label)
        leaf.is_leaf = True
        return leaf

    def choose_attribute(self, df, target_attr):
        information_gains = {}
        for attr in df.columns[:-1]:
            information_gains[attr] = self.get_information_gain(df, attr, target_attr)
        best_attr = max(information_gains, key=information_gains.get)
        return best_attr  # True if df[best_attr].dtype == "float64" else False

    def learn_decision_tree(self, examples, attributes, parent_examples):
        # check if examples is empty
        if examples.empty:
            return self.plurality_value(parent_examples)

        # if all examples have the same classification return it
        if examples["class"].nunique() == 1:
            leaf = Node(label=examples["class"].iloc[0])
            leaf.is_leaf = True
            return leaf
        # elif attributes is empty
        if len(attributes) == 1:
            return self.plurality_value(examples)

        # A <- argmax(a € attributes)IMPORTANCE(a, examples)
        best_attr = self.choose_attribute(examples, target_attr)
        decision_node = Node(attribute=best_attr)
        for value in examples[best_attr].unique():
            subset_df = examples[examples[best_attr] == value].drop(columns=best_attr)
            # recursively build the subtree using the subset of the dataset
            child_node = self.learn_decision_tree(
                subset_df, list(subset_df.columns), examples
            )
            # print(df.columns[value])
            decision_node.add_child(value, child_node)

        return decision_node
        # TLDR if discrete value we're gonna use information gain to choose the best attribute to split on

    def get_entropy(self, df, target_attr):
        entropy = 0
        value_counts = df[target_attr].value_counts()
        for value in value_counts:
            proportion = value / len(df[target_attr])
            entropy -= proportion * (math.log(proportion, 2))
        return entropy

    def get_information_gain(self, df, attr, target_attr):
        info_gain = self.get_entropy(df, target_attr)
        df_attr_counts = df[attr].value_counts()
        for value in df[attr].unique():
            subset = df[df[attr] == value]
            proportion = df_attr_counts[value] / len(df[attr])
            info_gain -= proportion * self.get_entropy(subset, target_attr)
        return info_gain


def print_tree(node, data, attribute_names=None, indent=""):
    if node.is_leaf:
        print(f"{indent}Leaf class: {node.label}")
        return
    else:
        print(f"{indent}{node.attribute}:")
    for value, child_node in node.children.items():
        print(f"{indent*5}  {value} -> ", end="")
        print_tree(child_node, data, attribute_names, indent + "    ")


"""REFACTORING NECESSARIO"""


def get_class_counts(df, target_attr):
    class_counts = {}
    for value in df[target_attr]:
        if value not in class_counts:
            class_counts[value] = 0
        class_counts[value] += 1
    return class_counts


def classify(instance, tree, data):
    # Traverse the tree until a leaf node is reached
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
