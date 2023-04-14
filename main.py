from DecisionTree import *
from helper import *
from crossValidation import *
from sklearn.model_selection import train_test_split

opz = int(input("Scegli il dataset che vuoi usare tra 1,2,3: "))

if opz == 1:
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    continuous_attrs = [0, 1, 2, 3]
    X, y, df = load_data(path, True, continuous_attrs)
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
elif opz == 2:
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data"
    X, y, df = load_data(path, False)
    df.columns = [
        "animal_name",
        "hair",
        "feathers",
        "eggs",
        "milk",
        "airbone",
        "aquatic",
        "predator",
        "toothed",
        "backbone",
        "breathes",
        "venomous",
        "fins",
        "legs",
        "tail",
        "domestic",
        "catsize",
        "class",
    ]
    df = df.drop("animal_name", axis=1)
elif opz == 3:
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00244/fertility_Diagnosis.txt"
    continuous_attrs = [1, 6, 8]
    X, y, df = load_data(path, True, continuous_attrs)
    df.columns = [
        "season_analysis_performed",
        "age_volunteer",
        "childish_diseases",
        "accident",
        "surgical_intervention",
        "high_fevers",
        "alcohol_consumption",
        "smoking",
        "hours_sitting",
        "class",
    ]

Decision = DecisionTree(df, list(df.columns))
#print_tree(Decision.tree, df, list(df.columns))
nodes, edges = bfs(Decision.tree)

dot = graphviz.Digraph(comment="BFS Tree")
for node in nodes:
    label = f"{node.attribute}" if node.attribute else f"{node.label}"
    shape = "ellipse" if node.is_leaf else "box"
    color = "green" if node.is_leaf else "lightblue"
    dot.node(str(node), label=label, shape=shape, color=color)
for edge in edges:
    dot.edge(str(edge[0]), str(edge[1]), label=str(edge[2]))

dot.render("./img/bfs_tree.gv", format="png")
print("Cross validation error: ", cross_validation(df, 10))
