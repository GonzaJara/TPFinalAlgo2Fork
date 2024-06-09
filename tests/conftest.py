import pytest
from decision_trees.models.tree_models import DecisionTreeClassifier
import pandas

@pytest.fixture(scope="session")
def overFittedTree():
    df = pandas.read_csv("play_tennis.csv")
    X = df.drop("play", axis=1)
    Y = df["play"]
    tree = DecisionTreeClassifier()
    return tree.fit(X, Y)