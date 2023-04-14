import pandas as pd
import math
import numpy as np

target_attr = "class"


class Node:
    def __init__(self, attribute=None, label=None):
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
        # this case is added by me, if we have only 2 examples left we might be left with an unoptimal split
        # it could happen that these 2 examples have identical values but different class and in that case
        # the split won't be optimal and can lead to overfitting
        if len(examples) == 2:
            return self.plurality_value(parent_examples)

        # if all examples have the same classification return it
        if len(examples["class"].unique()) == 1:
            leaf = Node(label=examples["class"].iloc[0])
            leaf.is_leaf = True
            return leaf
        # elif attributes is empty
        if len(attributes) == 1:
            return self.plurality_value(examples)

        # A <- argmax(a € attributes)IMPORTANCE(a, examples)
        best_attr = self.choose_attribute(examples, target_attr)
        # print(best_attr)
        decision_node = Node(attribute=best_attr)
        for value in examples[best_attr].unique():
            # print(examples[best_attr], value)
            subset_df = examples[examples[best_attr] == value].drop(columns=best_attr)
            # print("subset", subset_df)
            # recursively build the subtree using the subset of the dataset
            child_node = self.learn_decision_tree(
                subset_df, list(subset_df.columns), examples
            )
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
