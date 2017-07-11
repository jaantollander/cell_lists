import threading

import hypothesis.strategies as st
import numba
import numpy as np
from hypothesis.core import given, example
from hypothesis.extra.numpy import arrays

from cell_lists.cell_lists import add_to_cells, neighboring_cells, \
    iter_nearest_neighbors, partition_cells


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


@numba.jit(nopython=True, nogil=True, cache=True)
def brute_force(indices, points, radius):
    result = []
    for k, i in enumerate(indices[:-1]):
        for j in indices[k + 1:]:
            if np.linalg.norm(points[i, :] - points[j, :]) <= radius:
                result.append((i, j))
                result.append((j, i))
    return result


@numba.jit(nopython=True, nogil=True, cache=True)
def find_neighbors(cell_indices, neigh_cells, points_indices, cells_count,
                   cells_offset):
    result = []
    for i, j in iter_nearest_neighbors(cell_indices, neigh_cells,
                                       points_indices,
                                       cells_count, cells_offset):
        result.append((i, j))
    return result


def neighbor_distance_condition(cell_size, dimension, p0, p1):
    max_distance = 2 * cell_size * np.sqrt(dimension)
    distance = np.linalg.norm(p0 - p1)
    return distance <= max_distance or \
           np.isclose(distance, max_distance, rtol=1.e-3, atol=1.e-5)


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

    correct = set(brute_force(np.arange(size), points, cell_size))
    result = find_neighbors(cell_indices, neigh_cells, points_indices,
                            cells_count, cells_offset)

    for i, j in result:
        assert neighbor_distance_condition(
            cell_size, dimension, points[i, :], points[j, :])

    results_set = {(i, j) for i, j in result if
                   np.linalg.norm(points[i, :] - points[j, :]) <= cell_size}

    assert results_set.issubset(correct)


def find_neighbors_thread(index, results, cell_indices, neigh_cells,
                          points_indices, cells_count, cells_offset):
    results[index].extend(
        find_neighbors(cell_indices, neigh_cells, points_indices, cells_count,
                       cells_offset))


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
    splits = partition_cells(n, cells_count, neigh_cells)

    cell_indices_chucks = (cell_indices[start:end] for start, end in
                           zip(splits[:-1], splits[1:]))

    # TODO: save results and compare with single threaded version
    # TODO: check that the solution is valid
    results = [[] for _ in range(n)]

    # Spawn one thread per chunk
    threads = [threading.Thread(
        target=find_neighbors_thread,
        args=(i, results, chunk, neigh_cells, points_indices, cells_count,
              cells_offset))
        for i, chunk in enumerate(cell_indices_chucks)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    res = sum(results, [])
    correct = find_neighbors(cell_indices, neigh_cells, points_indices,
                             cells_count, cells_offset)

    assert res == correct
