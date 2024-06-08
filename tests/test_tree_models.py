import pytest
import numpy as np
import random

# with pytest.raises(ValueError):
#     some_function_raises_ValueError(*args)

def test_baseTree(decisionTreeMaxDepth4):
    assert decisionTreeMaxDepth4.get_depth() <= 4
    