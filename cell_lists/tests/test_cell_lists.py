import threading

import hypothesis.strategies as st
import numba
import numpy as np
from hypothesis.core import given, example
from hypothesis.extra.numpy import arrays

from cell_lists.cell_lists import add_to_cells, neighboring_cells, \
    find_neighbors, split_into_parts


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
    """Test that add to cells returns correct values."""
    index_list, count, offset, shape = add_to_cells(points, cell_size)

    size, dimensions = points.shape
    for i in range(size):
        assert i in index_list
    assert np.sum(count) == size
    assert 0 <= np.min(offset) <= np.max(offset) <= size
    assert np.all(np.sort(offset) == offset)


@given(arrays(dtype=np.int64, shape=1) |
       arrays(dtype=np.int64, shape=2) |
       arrays(dtype=np.int64, shape=3))
def test_neighboring_cells(grid_shape):
    """Test that neighboring cells works."""
    neigh = neighboring_cells(grid_shape)
    assert True


@given(points=reals(-10.0, 10.0, shape=(10, 2)) |
              reals(-10.0, 10.0, shape=(10, 3)),
       cell_size=st.floats(0.1, 1.0))
def test_find_neighbors(points, cell_size):
    """Test that neighbors are withing the correct distance from each other

    .. math::
       d(\mathbf{p}_i, \mathbf{p}_j) \leq \sqrt{n (2 c)^2} = 2c \sqrt{n}

    """
    size, dimension = points.shape
    points_indices, cells_count, cells_offset, grid_shape = add_to_cells(
        points, cell_size)
    cell_indices = np.arange(len(cells_count))
    neigh_cells = neighboring_cells(grid_shape)

    for i, j in find_neighbors(cell_indices, neigh_cells, points_indices,
                               cells_count, cells_offset):
        max_distance = 2 * cell_size * np.sqrt(dimension)
        distance = np.linalg.norm(points[i, :] - points[j, :])
        assert distance <= max_distance or \
               np.isclose(distance, max_distance, rtol=1.e-3, atol=1.e-5)


@numba.jit(nopython=True, nogil=True, cache=True)
def consume_find_neighbors(cell_indices, neigh_cells, points_indices,
                           cells_count, cells_offset):
    for _ in find_neighbors(cell_indices, neigh_cells, points_indices,
                            cells_count, cells_offset):
        pass


def test_multithreaded():
    n = 4
    low = -1.0
    high = 1.0
    size = 1000
    points = np.random.uniform(low, high, size=(size, 2))
    cell_size = 0.01

    points_indices, cells_count, cells_offset, grid_shape = add_to_cells(
        points, cell_size)
    cell_indices = np.arange(len(cells_count))
    neigh_cells = neighboring_cells(grid_shape)
    splits = split_into_parts(n, cells_count, neigh_cells)

    cell_indices_chucks = (
        cell_indices[start:end] for start, end in zip(splits[:-1], splits[1:]))

    # Spawn one thread per chunk
    threads = [threading.Thread(
        target=consume_find_neighbors,
        args=(chunk, neigh_cells, points_indices, cells_count, cells_offset))
        for chunk in cell_indices_chucks]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert True
