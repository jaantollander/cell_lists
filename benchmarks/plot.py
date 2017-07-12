from bokeh.plotting import figure, show, output_file

from benchmarks.benchmark import benchmark_add_to_cells, \
    benchmark_find_neighbors, benchmark_split_into_parts, benchmark_brute_force


# TODO: color cycles
# TODO: add legend
# TODO: transform axes labels


def plot(*dataframes):
    p = figure(title="Benchmarks", plot_height=600, plot_width=1200)
    output_file("benchmarks.html", title="Benchmarks")
    for df in dataframes:
        mean = df['mean']
        p.circle(mean.index, mean.data)
        p.line(mean.index, mean.data)
    show(p)


points_range = list(range(100, 10000, 100))
cell_size = 0.01
plot(
    # benchmark_add_to_cells(points_range, cell_size),
    # benchmark_split_into_parts(points_range, cell_size, 3),
    benchmark_find_neighbors(points_range, cell_size, num_threads=1),
    benchmark_find_neighbors(points_range, cell_size, num_threads=2),
    # benchmark_find_neighbors(points_range, cell_size, num_threads=3),
    # benchmark_brute_force(points_range, cell_size, iterations=10),
)
