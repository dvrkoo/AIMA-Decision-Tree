from os import name
import pandas as pd
from sklearn import datasets


class DataSet:
    def __init__(self, name, attr_names) -> None:
        self.name = name
        self.data = self.open_data()
        self.data.columns = attr_names
        self.attr_names = attr_names
        self.attr_dict = self.get_attr_dict(attr_names)

    def open_data(self):
        return pd.read_csv("./" + self.name + ".data")

    def get_attr_dict(self, attr_names):
        attr_dict = {}
        for attr in attr_names:
            attr_dict[attr] = self.data[attr].unique().tolist()
        return attr_dict

    def factorizer(self, name):
        self.data[name] = pd.factorize(self.data["class"])[0]


def iris_builder():
    return DataSet(
        "iris",
        attr_names=[
            "sepal_length",
            "sepal_width",
            "petal_lenght",
            "petal_width",
            "class",
        ],
    )

""""
def main():
    iris = iris_builder()
    iris.factorizer("class")
    print(iris.data)


main()"""
