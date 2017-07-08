import numba
import numpy as np
from numba import f8, i8

NEIGHBOR_INDICES = ((1, 0), (1, 1), (0, 1), (1, -1))


@numba.jit([(f8[:, :], f8)], nopython=True, nogil=True, cache=True)
def add_to_bins(points, cell_size):
    """
    Cell lists algorithm partitions space into squares and sorts points into
    the square they belong. This allows fast neighbourhood search because we
    only have to search current and neighbouring cells for points.

    Args:
        points (numpy.ndarray):
            Array of :math:`N` points :math:`(\mathbf{p}_i \in
            \mathbb{R}^2)_{i=1,...,N}` (``shape=(size, 2)``) to be block listed

        cell_size (float):
            Positive real number :math:`c > 0`. Width and height of the
            rectangular mesh.

    Returns:
        (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray):

    """
    assert cell_size > 0 and points.ndim == 2 and points.shape[1] == 2

    # Dimensions (rows, columns)
    n, m = points.shape

    # Compute index ranges for indices
    x_min = (points[0, :] / cell_size).astype(np.int64)
    x_max = (points[0, :] / cell_size).astype(np.int64)
    for i in range(1, n):
        for j in range(m):
            x = np.int64(points[i, j] / cell_size)
            if x < x_min[j]:
                x_min[j] = x
            if x > x_max[j]:
                x_max[j] = x

    # Blocks
    indices = np.zeros(shape=points.shape, dtype=np.int64)
    for i in range(n):
        for j in range(m):
            indices[i, j] = np.int64(points[i, j] / cell_size) - x_min[j]

    shape = (x_max - x_min) + 1

    # Count how many points go into each cell
    size = np.prod(shape)
    count = np.zeros(size, dtype=np.int64)
    for i in range(n):
        index = indices[i, 0]
        for j in range(1, m):
            index *= shape[j]
            index += indices[i, j]
        count[index] += 1

    # Index list
    index_list = np.zeros(n, dtype=np.int64)
    offset = count.cumsum() - 1  # Offset indices
    for i in range(n):
        index = indices[i, 0]
        for j in range(1, m):
            index *= shape[j]
            index += indices[i, j]
        index_list[offset[index]] = i
        offset[index] -= 1

    offset += 1

    return index_list, count, offset, shape


@numba.jit([i8[:](i8[:], i8[:], i8[:], i8[:], i8[:])],
           nopython=True, nogil=True, cache=True)
def find_cell(indices, index_list, count, offset, shape):
    r"""Multidimensional indexing

    Args:
        indices (numpy.ndarray | tuple):

    Returns:
        numpy.ndarray:
    """
    # TODO: Handle index out of bound
    index = indices[0]
    for j in range(1, len(indices)):
        index *= shape[j]
        index += indices[j]
    start = offset[index]
    end = start + count[index]
    return index_list[start:end]


@numba.jit([(i8[:], i8[:], i8[:], i8[:])], nopython=True, nogil=True)
def find_neighbors(index_list, count, offset, shape):
    r"""Iterate over cell lists

    Args:
        index_list:
        count:
        offset:
        shape:

    Yields:
        (int, int):

    """
    n, m = shape

    for i in range(n):
        for j in range(m):
            # Herding between agents inside the block
            ilist = find_cell(np.array((i, j)), index_list, count, offset, shape)
            for l, i_agent in enumerate(ilist[:-1]):
                for j_agent in ilist[l + 1:]:
                    yield i_agent, j_agent

            # Herding between agent inside the block and neighbouring agents
            for k in range(len(NEIGHBOR_INDICES)):
                i2, j2 = NEIGHBOR_INDICES[k]
                if 0 <= (i + i2) < n and 0 <= (j + j2) < m:
                    ilist2 = find_cell(np.array((i + i2, j + j2)), index_list,
                                       count, offset, shape)
                    for i_agent in ilist:
                        for j_agent in ilist2:
                            yield i_agent, j_agent
