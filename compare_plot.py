from bokeh.layouts import column, row, layout, gridplot
from bokeh.models import ColumnDataSource, CategoricalTicker, Div
from bokeh.plotting import figure, output_file, show
from bokeh.sampledata.autompg import autompg
from bokeh.transform import jitter
from bokeh.palettes import Spectral6
from bokeh.models import Legend
from collections import defaultdict

import argparse
import json
import pandas as pd
import sys
import re
from tools.data import load_data_dir, load_data_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare_files", nargs='+',
                        help='json data files formatted by pytest-benchmark module')
    parser.add_argument("--benchmark_names", nargs='+',
                        help="names of benchmarks expected to exist in each of the"
                             " provided json files")
    parser.add_argument("--data_dir", nargs='+')
    parser.add_argument("--most_recent_n", default=10, type=int,
                        help="plot only the most recent data files in data_dir")

    parser.add_argument("--output_html", default='plot.html', help="html file to write")
    args = parser.parse_args()
    plot_height = 300
    plot_width = 400

    if args.data_dir is not None:
        if len(args.data_dir) > 1:
            compare_datasets = {d: load_data_dir(d, most_recent_files=-1) for d in args.data_dir}
        else:
            data = load_data_dir(args.data_dir, most_recent_files=args.most_recent_n)
    elif args.compare_files is None or len(args.compare_files) == 0:
        print("Must provide at least one json file")
        sys.exit(-1)
    else:
        data = load_data_files(args.compare_files)

    if compare_datasets:
        all_names = set()
        ntags = 0
        tags = {}
        for dataset in compare_datasets:
            data = compare_datasets[dataset]
            all_names.update(data.benchmark_names(keyword_filter=args.benchmark_names))
            tags[dataset] = data.tags()
            ntags = max(ntags, len(tags[dataset]))
        
        plots = []
        tags = [str(i) for i in range(ntags)]
        colors = Spectral6

        regexp = "test_(.*)\[(.*)-(.*)-(.*)\]"
        model_names = set([re.search(regexp, x).groups()[1] for x in all_names])
        column_groups = (('cpu', 'eager'), ('cpu', 'jit'), ('cuda', 'eager'), ('cuda', 'jit'))
        row_plots = defaultdict(dict)
        for name in all_names:
            test, model, device, compiler = re.search(regexp, name).groups()
            p = figure(x_range=tags, title=f"{device}, {compiler}", plot_height=plot_height, plot_width=plot_width)
            p.xgrid.grid_line_color = None
            # p.xaxis.major_label_orientation = "vertical"
            legend_items = []
            for i, dataset in enumerate(compare_datasets):

                data = compare_datasets[dataset]
                series = p.circle(x='file_idx', y='time', source=data.as_dataframe(name), color=colors[i])
                legend_items.append((dataset, [series,] ))
            legend = Legend(items=legend_items, location=(0,0))
            legend.click_policy = 'hide'
            # p.add_layout(legend, 'above')
            # plots.append(p)
            row_plots[(test, model)][(device, compiler)] = p


        plots = []
        for name in row_plots:
            plots.append(layout([
                [Div(text=f"<h3>test_{name[0]}[{name[1]}]</h3>")],
                gridplot([row_plots[name].get(col, Div(width=plot_width, height=plot_height)) for col in column_groups],
                         ncols=len(column_groups),
                         plot_height=plot_height, plot_width=plot_width)
            ]))
    
    else:
        names = data.benchmark_names(keyword_filter=args.benchmark_names)
        tags = data.tags()
        plots = []
        for name in names:
            p = figure(x_range=tags, title=name)
            p.xgrid.grid_line_color = None
            # p.xaxis.major_label_orientation = "vertical"
            p.circle(x='tag', y='time', source=data.as_dataframe(name))
            plots.append(p)
    
    output_file(args.output_html)
    show(column(*plots))


    