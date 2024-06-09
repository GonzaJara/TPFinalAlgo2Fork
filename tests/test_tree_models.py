import pytest
import numpy as np
# with pytest.raises(ValueError):
#     some_function_raises_ValueError(*args)

def test_baseTree(overFittedTree):
    assert overFittedTree.get_depth() == 5