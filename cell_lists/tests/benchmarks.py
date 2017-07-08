import pytest
import numpy as np

from cell_lists.cell_lists import add_to_cells


low = -1.0
high = 1.0
cell_sizes = (0.01, 0.05, 0.1)
sizes = (100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000)


@pytest.mark.parametrize('size', sizes)
@pytest.mark.parametrize('cell_size', cell_sizes)
def benchmark_add_to_cells(benchmark, cell_size, size):
    points = np.random.uniform(low, high, size=(size, 2))
    benchmark(add_to_cells, points, cell_size)
    assert True
