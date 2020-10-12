from bokeh.layouts import column, row, layout, gridplot
from bokeh.plotting import figure, output_file, show
from bokeh.sampledata.autompg import autompg
from bokeh.transform import jitter
from bokeh.palettes import Category10
from bokeh.models import Legend
from bokeh.models import ColumnDataSource, CategoricalTicker, Div
from bokeh.models import ColumnDataSource, DataTable, DateFormatter, TableColumn
from bokeh.transform import jitter
from numpy import linspace
from scipy.stats import gaussian_kde

import argparse
import json
import pandas as pd
import os
import sys
import re
from collections import defaultdict
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
    plot_height = 125
    plot_width = 225

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

    # TODO
    # Fix y axis 0
    # fix alpha
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
        tags = [i for i in range(ntags)]
        colors = Category10[10]
        regexp = "test_(.*)\[(.*)-(.*)-(.*)\]"
        model_names = set([re.search(regexp, x).groups()[1] for x in all_names])

        doc_title = Div(text="<h1>TorchBenchmark Comparison Plot</h1>")

        # uberlegend_data = dict(
        #         tags=tags,
        #     )
        # uberlegend_columns = [
        #         TableColumn(field="tags", title="Tag"),
        #     ]
        # for dataset in compare_datasets:
        #     dataset_tags = compare_datasets[dataset].tags()
        #     dataset_tags = [os.path.relpath(fullname, dataset) for fullname in dataset_tags]
        #     uberlegend_data[dataset] = dataset_tags + [""] * (ntags - len(dataset_tags))
        #     uberlegend_columns.append(TableColumn(field=dataset, title=dataset))
        # uberlegend_source = ColumnDataSource(uberlegend_data)
        # uberlegend_title = Div(text="<h2>Datasets in this benchmark</h2>")
        # uberlegend = DataTable(source=uberlegend_source, columns=uberlegend_columns, height=200, width=2000)
        color_legend = [[Div(text="<h3>Legend</h3>")]]
        for i, dataset in enumerate(compare_datasets):
            color_legend.append([Div(text=f"{dataset}:"), Div(text="---", style={'color': colors[i], 'background-color': colors[i]})])
        color_legend = gridplot(color_legend)

        column_groups = [(a, b, c) for a in ('train', 'eval') for b in ('cpu', 'cuda') for c in ('eager', 'jit')]
        row_plots = defaultdict(dict)
        legends = {}
        for name in all_names:
            test, model, device, compiler = re.search(regexp, name).groups()
            p = figure(title=f"{device}, {compiler}",
                       plot_height=plot_height, plot_width=plot_width,
                       x_axis_label='Time (S)', y_axis_label='KDE')
            p.yaxis.visible = False
            p.xgrid.grid_line_color = None
            p.xaxis.minor_tick_line_color = None
            p.xaxis.ticker.desired_num_ticks = 3
            p.ygrid.grid_line_color = None
            legend_items = []
            for i, dataset in enumerate(compare_datasets):
                data = compare_datasets[dataset].as_dataframe(name, max_data=-1)
                xdata = linspace(0, data.time.max() * 1.25, 125)
                dataset_series = []
                for idx in tags:
                    if len(data[data.file_idx==idx]) > 0:

                        pdf = gaussian_kde(data[data.file_idx==idx].time)
                        series = p.line(y=pdf(xdata), x=xdata, color=colors[i])
                                        # legend_label=dataset)
                        # p.legend.location = 'top_left'
                        # p.legend.label_text_font_size = '8pt'
                        dataset_series.append(series)
                legend_items.append((dataset, dataset_series))
            l = Legend(items=legend_items)
            l.click_policy = 'hide'
            # p.add_layout(l, 'right')j
            row_plots[model][(test, device, compiler)] = p

            p.y_range.start = 0


        plots = [doc_title, color_legend] #uberlegend_title, uberlegend]
        for model in sorted(row_plots):

            gp = gridplot([[Div(text='train')] + [row_plots[model].get(col, Div(width=plot_width, height=plot_height)) for col in column_groups if 'train' in col],
                           [Div(text='eval')] + [row_plots[model].get(col, Div(width=plot_width, height=plot_height)) for col in column_groups if 'eval' in col],
                        ],
                        plot_height=plot_height, plot_width=plot_width)
            plots.append(layout([
                [Div(text=f"<h3>{model}</h3>")],
                gp,
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


    