from bokeh.plotting import figure, show, output_file

from benchmarks.benchmark import benchmark_add_to_cells, \
    benchmark_find_neighbors, benchmark_split_into_parts


def plot(*dataframes):
    # TODO: color cycles
    # TODO: add legend
    # TODO: transform axes labels

    p = figure(title="Benchmarks")
    output_file("benchmarks.html", title="Benchmarks")
    for df in dataframes:
        d = df['mean']
        p.circle(d.index, d.data)
        p.line(d.index, d.data)
    show(p)


plot(
    benchmark_add_to_cells(range(100, 1000, 100), 0.01),
    # benchmark_split_into_parts(range(100, 1000, 100), 0.01, 2),
    benchmark_split_into_parts(range(100, 1000, 100), 0.01, 3),
    benchmark_find_neighbors(range(100, 1000, 100), 0.01, 1)
)
