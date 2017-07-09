import numba
import numpy as np
import pytest

from cell_lists.cell_lists import add_to_cells, find_neighbors, \
    neighboring_cells

low = -1.0
high = 1.0
cell_sizes = (0.01, 0.05, 0.1)
sizes = (100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000)


@numba.jit(nopython=True, nogil=True, cache=True)
def consume_find_neighbors(cell_indices, neigh_cells, points_indices,
                           cells_count, cells_offset):
    for _ in find_neighbors(cell_indices, neigh_cells, points_indices,
                            cells_count, cells_offset):
        pass


@pytest.mark.parametrize('size', sizes)
@pytest.mark.parametrize('cell_size', cell_sizes)
def benchmark_add_to_cells(benchmark, cell_size, size):
    points = np.random.uniform(low, high, size=(size, 2))
    benchmark(add_to_cells, points, cell_size)
    assert True


@pytest.mark.parametrize('size', sizes)
@pytest.mark.parametrize('cell_size', cell_sizes)
def benchmark_find_neighbors(benchmark, cell_size, size):
    points = np.random.uniform(low, high, size=(size, 2))
    points_indices, cells_count, cells_offset, grid_shape = add_to_cells(
        points, cell_size)
    cell_indices = np.arange(len(cells_count))
    neigh_cells = neighboring_cells(grid_shape)
    benchmark(consume_find_neighbors, cell_indices, neigh_cells, points_indices,
              cells_count, cells_offset)
    assert True
