import threading
from timeit import default_timer as timer

import numba
import numpy as np
import pandas as pd

from cell_lists.core import add_to_cells, iter_nearest_neighbors, \
    neighboring_cells, partition_cells


# TODO: add warmup iterations
# TODO: test distance between neighbors and add some work to test partitioning
# TODO: progressbar
# TODO: memory usage
# TODO: add metadata to dataframes


@numba.jit(['(i8[:], f8[:, :], f8)'], nopython=True, nogil=True, cache=True)
def brute_force(indices, points, cutoff):
    for k, i in enumerate(indices[:-1]):
        for j in indices[k + 1:]:
            if np.sum((points[i, :] - points[j, :]) ** 2) < cutoff**2:
                pass


@numba.jit(['(i8[:], i8[:], i8[:], i8[:], i8[:], f8[:, :], f8)'],
           nopython=True, nogil=True, cache=True)
def consume_find_neighbors(cell_indices, neigh_cells, points_indices,
                           cells_count, cells_offset, points, cutoff):
    for i, j in iter_nearest_neighbors(
            cell_indices, neigh_cells, points_indices, cells_count,
            cells_offset):
        if np.sum((points[i, :] - points[j, :]) ** 2) < cutoff ** 2:
            pass


def consume_find_neighbors_multithread(
        n, cell_indices, neigh_cells, points_indices, cells_count,
        cells_offset, points, cutoff):
    splits = partition_cells(n, cells_count, neigh_cells)
    cell_indices_chucks = (cell_indices[start:end] for start, end in
                           zip(splits[:-1], splits[1:]))

    # Spawn one thread per chunk
    threads = [threading.Thread(
        target=consume_find_neighbors,
        args=(chunk, neigh_cells, points_indices, cells_count, cells_offset,
              points, cutoff))
        for chunk in cell_indices_chucks]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def time(f, *args, **kwargs):
    start = timer()
    ret = f(*args, **kwargs)
    end = timer()
    return end - start


def benchmark_add_to_cells(points_range, cell_size, dimensions=2, low=0.0,
                           high=1.0, iterations=1000):
    df = pd.DataFrame(np.zeros((len(points_range), 1)),
                      index=points_range, columns=('mean',))
    df.index.name = 'points_size'
    df.columns.name = 'stats'

    for point_size in points_range:
        times = np.zeros(iterations)

        for i in range(iterations):
            points = np.random.uniform(low, high,
                                       size=(point_size, dimensions))
            t = time(add_to_cells, points, cell_size)
            times[i] = t
        df['mean'][point_size] = np.mean(times)

    return df


def benchmark_split_into_parts(points_range, cell_size, num_threads,
                               dimensions=2, low=0.0, high=1.0,
                               iterations=1000):
    df = pd.DataFrame(np.zeros((len(points_range), 1)),
                      index=points_range, columns=('mean',))
    df.index.name = 'points_size'
    df.columns.name = 'stats'

    for size in points_range:
        times = np.zeros(iterations)

        for i in range(iterations):
            points = np.random.uniform(low, high, size=(size, dimensions))
            points_indices, cells_count, cells_offset, grid_shape = add_to_cells(
                points, float(cell_size))
            neigh_cells = neighboring_cells(grid_shape)
            t = time(partition_cells, num_threads, cells_count, neigh_cells)
            times[i] = t

        df['mean'][size] = np.mean(times)

    return df


def benchmark_find_neighbors(points_range, cell_size, num_threads, dimensions=2,
                             low=0.0, high=1.0, iterations=100):
    df = pd.DataFrame(np.zeros((len(points_range), 1)),
                      index=points_range, columns=('mean',))
    df.index.name = 'points_size'
    df.columns.name = 'stats'

    for size in points_range:
        times = np.zeros(iterations)

        for i in range(iterations):
            points = np.random.uniform(low, high, size=(size, dimensions))
            points_indices, cells_count, cells_offset, grid_shape = add_to_cells(
                points, cell_size)
            cell_indices = np.arange(len(cells_count))
            neigh_cells = neighboring_cells(grid_shape)

            if num_threads == 1:
                t = time(consume_find_neighbors,
                         cell_indices, neigh_cells, points_indices,
                         cells_count, cells_offset, points, cell_size)
            elif num_threads > 1:
                t = time(consume_find_neighbors_multithread,
                         num_threads, cell_indices, neigh_cells, points_indices,
                         cells_count, cells_offset, points, cell_size)
            else:
                raise ValueError

            times[i] = t

        df['mean'][size] = np.mean(times)

    return df


def benchmark_brute_force(points_range, cell_size, dimensions=2, low=0.0,
                          high=1.0, iterations=100):
    df = pd.DataFrame(np.zeros((len(points_range), 1)),
                      index=points_range, columns=('mean',))
    df.index.name = 'points_size'
    df.columns.name = 'stats'

    for size in points_range:
        times = np.zeros(iterations)

        for i in range(iterations):
            points = np.random.uniform(low, high, size=(size, dimensions))
            indices = np.arange(len(points))
            times[i] = time(brute_force, indices, points, cell_size)

        df['mean'][size] = np.mean(times)

    return df
