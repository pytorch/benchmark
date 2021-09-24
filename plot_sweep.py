import argparse
import json
# import pandas as pd
import os
# import sys
# import re
import yaml
import itertools

# from bokeh.layouts import column, row, layout, gridplot
# from bokeh.plotting import figure, output_file, show
# from bokeh.sampledata.autompg import autompg
# from bokeh.transform import jitter
from bokeh.palettes import Category10
from bokeh.models import HoverTool, Div, Range1d, HoverTool
from bokeh.plotting import figure, output_file, show
# from bokeh.models import Legend
# from bokeh.models import ColumnDataSource, CategoricalTicker, Div
# from bokeh.models import ColumnDataSource, DataTable, DateFormatter, TableColumn
# from bokeh.transform import jitter
from collections import defaultdict
from datetime import datetime as dt
from torchbenchmark.util.data import load_data_dir, load_data_files
from torchbenchmark.score.compute_score import TorchBenchScore

TORCHBENCH_SCORE_VERSION = "v1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", nargs='+',
                        help="One or more directories containing benchmark json files. "
                             "Each directory will be plotted as a separate series. "
                             "By default, the first file in the first directory will be used"
                             " to generate a score configuration with a target of 1000,"
                             " and everything else will be relative to that.")
    parser.add_argument("--output_html", default='plot.html', help="html file to write")
    parser.add_argument("--plot_all", action='store_true',
                        help="Plots the scores for each configuration")
    parser.add_argument("--reference_json", required=True,
                        help="file defining score norm values, usually first json in first data_dir")

    args = parser.parse_args()
    plot_height = 800
    plot_width = 1000

    assert len(args.data_dir) > 0, "Must provide at least one data directory"
    compare_datasets = [load_data_dir(d, most_recent_files=-1) for d in args.data_dir]

    with open(args.reference_json) as f:
        ref_data = json.load(f)
    plot_all = args.plot_all
    score_config = TorchBenchScore(ref_data=ref_data, version=TORCHBENCH_SCORE_VERSION)

    p = figure(plot_width=plot_width, plot_height=plot_height,
               x_axis_type='datetime')

    xs = []
    ys = []
    zs = []
    max_score = 0
    for d in compare_datasets:
        scores = {}
        scores_db = defaultdict(list)
        for i in range(len(d._json_raw)):
            data = d._json_raw[i]
            pytorch_ver = data['machine_info']['pytorch_version']
            # Slice the portion after '+'
            pytorch_ver_cuda_loc = pytorch_ver.rfind('+')
            pytorch_ver = pytorch_ver[:pytorch_ver_cuda_loc]
            date = dt.strptime(pytorch_ver[pytorch_ver.index("dev") + len("dev"):], "%Y%m%d")
            score = score_config.compute_score(data)
            scores[date] = score

        dates = []
        total_scores = []
        all_scores = []
        for date in sorted(scores.keys()):
            dates.append(date)
            total_scores.append(scores[date]["total"])
            max_score = max(max_score, max(total_scores))
            all_scores.append(scores[date])
        xs.append(dates)
        ys.append(total_scores)
        if plot_all:
            zs.append(all_scores)

    colors = itertools.cycle(Category10[10])
    basenames = map(os.path.basename, args.data_dir)

    if plot_all:
        for x, z in zip(xs, zs):
            basename = next(basenames)
            color = next(colors)
            configs = z[0].keys()
            for config in configs:
                if not ("subscore" in config or "total" in config):
                    continue
                color = next(colors)
                scores = []
                for s in z:
                    scores.append(s[config])
                p.line(x, scores, color=color, line_width=2, legend_label=basename + '-' + config)

        p.legend.click_policy = "hide"
    else:
        for x, y, color in zip(xs, ys, colors):
            p.line(x, y, color=color, line_width=2, legend_label=next(basenames))

        for x, y, color in zip(xs, ys, colors):
            p.circle(x, y, color=color)

    p.legend.location = "bottom_right"
    p.y_range = Range1d(0, max_score * 1.25)
    p.add_tools(HoverTool(
        tooltips=[
            ('date',    '@x{%F}'),
            ('score',   '@y{0.00 a}'),
        ],
        formatters={
            '@x':      'datetime',
            '@y':     'numeral',
        },
    ))
    output_file(args.output_html)
    show(p)
