import pandas as pd
import graphviz
from queue import Queue


def continue_to_categoric(examples, attribute):
    print("DATASET CONTAINS CONTINUOUS VALUES, APPLYING QUANTILES TO CATEGORIZE IT")
    print(examples)
    examples = pd.qcut(examples, 6, labels=False, duplicates="drop")
    print("CATEGORIZED THE DATASET")
    return examples


def load_data(file_path, is_continuous, continuous_attrs=[]):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    # Split the DataFrame into examples and targets
    if is_continuous:
        for el in continuous_attrs:
            df.iloc[:, el] = continue_to_categoric(df.iloc[:, el], el)
    examples = df.iloc[:, :0]
    targets = df.iloc[:, 0]

    # Return the examples and targets as numpy arrays
    return examples.to_numpy(), targets.to_numpy(), df


def print_tree(node, data, attribute_names=None, indent=""):
    if node.is_leaf:
        print(f"{indent}Leaf class: {node.label}")
        return
    else:
        print(f"{indent}{node.attribute}:")
    for value, child_node in node.children.items():
        print(f"{indent*5}  {value} -> ", end="")
        print_tree(child_node, data, attribute_names, indent + "    ")


def bfs(root):
    queue = Queue()
    queue.put(root)
    nodes = []
    edges = []
    while not queue.empty():
        node = queue.get()
        nodes.append(node)
        for value, child_node in node.children.items():
            edges.append((node, child_node, value))
            queue.put(child_node)
    return nodes, edges
