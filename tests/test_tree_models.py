import pytest
import numpy as np
# with pytest.raises(ValueError):
#     some_function_raises_ValueError(*args)

def test_imports():
    from decision_trees import BaseTree
    from decision_trees import CategoricDecision
    from decision_trees import RandomForestClassifier
    from decision_trees import DecisionTreeClassifier

def test_baseTree(overFittedTree):
    assert overFittedTree.get_depth() == 5
    assert overFittedTree.get_depth() == 5
    assert overFittedTree.get_depth() == 5
    assert overFittedTree.get_depth() == 5
    assert overFittedTree.get_depth() == 5
    assert overFittedTree.get_depth() == 5
    assert overFittedTree.get_depth() == 5
    assert overFittedTree.get_depth() == 5
    assert overFittedTree.get_depth() == 5
    assert overFittedTree.get_depth() == 5