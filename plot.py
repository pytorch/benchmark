from bokeh.layouts import column
from bokeh.models import ColumnDataSource, CategoricalTicker
from bokeh.plotting import figure, output_file, show
from bokeh.sampledata.autompg import autompg
from bokeh.transform import jitter

import argparse
import json
import pandas as pd
import sys
from tools.data import load_data_dir, load_data_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare_files", nargs='+',
                        help='json data files formatted by pytest-benchmark module')
    parser.add_argument("--benchmark_names", nargs='+',
                        help="names of benchmarks expected to exist in each of the"
                             " provided json files")
    parser.add_argument("--benchmark_data_dir")

    parser.add_argument("--output_html", default='plot.html', help="html file to write")
    args = parser.parse_args()

    if args.compare_files is None or len(args.compare_files) == 0:
        print("Must provide at least one json file")
        sys.exit(-1)

    data = load_data_files(args.compare_files)
    names = data.benchmark_names(keyword_filter=args.benchmark_names)
    tags = args.compare_files
    plots = []
    for name in names:
        p = figure(x_range=tags, title=name)
        p.xgrid.grid_line_color = None
        # p.xaxis.major_label_orientation = "vertical"
        p.circle(x='tag', y='time', source=data.as_dataframe(name))
        plots.append(p)
    output_file(args.output_html)
    show(column(*plots))


    