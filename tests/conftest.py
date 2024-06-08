import pytest
from decision_trees.models.tree_models import DecisionTreeClassifier
import pandas
from sklearn.model_selection import train_test_split

@pytest.fixture(scope="session")
def decisionTreeNoParams():
    df = pandas.read_csv("play_tennis.csv")
    X = df.drop("play", axis=1)
    Y = df["play"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    tree = DecisionTreeClassifier()
    return tree.fit(X_train, Y_train)

@pytest.fixture(scope="session")
def decisionTreeMaxDepth4():
    df = pandas.read_csv("play_tennis.csv")
    X = df.drop("play", axis=1)
    Y = df["play"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    tree = DecisionTreeClassifier(max_depth=4)
    return tree.fit(X_train, Y_train)