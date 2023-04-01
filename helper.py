import pandas as pd


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
    if is_continuous:
        df.iloc[:, :-1] = continue_to_categoric(
            df.iloc[:, :-1], df.iloc[:, :-1].columns
        )
    examples = df.iloc[:, :-1]
    targets = df.iloc[:, -1]

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
