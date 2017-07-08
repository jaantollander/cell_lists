import numpy as np
from hypothesis.core import given, example
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays

from cell_lists.cell_lists import add_to_cells


def reals(min_value=None,
          max_value=None,
          exclude_zero=None,
          shape=None,
          dtype=np.float):
    """Real number strategy that excludes nan and inf.

    Args:
        min_value (Number, optional):
        max_value (Number, optional):
        exclude_zero (str, optional):
            Choices from: (None, 'exact', 'near')
        shape (int|tuple, optional):
            None for scalar output and int or tuple of int for array output.
        dtype (numpy.float):
            Numpy float type
    """
    # TODO: size as strategy
    assert dtype is None or np.dtype(dtype).kind == u'f'
    if min_value is not None and max_value is not None:
        assert max_value > min_value

    elements = st.floats(min_value, max_value, False, False)

    # Filter values
    if exclude_zero == 'exact':
        elements = elements.filter(lambda x: x != 0.0)
    elif exclude_zero == 'near':
        elements = elements.filter(lambda x: not np.isclose(x, 0.0))

    # Strategy
    if shape is None:
        return elements
    else:
        return arrays(dtype, shape, elements)


@given(points=reals(-10.0, 10.0, shape=(10, 2)),
       cell_size=st.floats(0.1, 1.0))
@example(points=np.zeros((0, 2)), cell_size=0.1)
@example(points=np.zeros((1, 2)), cell_size=0.1)
def test_add_to_cells(points, cell_size):
    index_list, count, offset, shape = add_to_cells(points, cell_size)

    size, dimensions = points.shape
    for i in range(size):
        assert i in index_list
    assert np.sum(count) == size
    assert 0 <= np.min(offset) <= np.max(offset) <= size
    assert np.all(np.sort(offset) == offset)


def test_find_neighbors():
    assert True
