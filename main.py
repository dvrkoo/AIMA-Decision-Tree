from DecisionTree import *


opz = int(input("Scegli il dataset che vuoi usare tra 1,2,3: "))

if opz == 1:
    X, y, df = load_data("iris.data", True)
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
elif opz == 2:
    X, y, df = load_data("zoo.data", False)
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
    X, y, df = load_data("fertility_diagnosis.txt", True)
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

print(df)
Decision = DecisionTree(df, list(df.columns))
print_tree(Decision.tree, df)
print("Cross validation error: ", cross_validation(df, 10))
