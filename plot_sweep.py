import argparse
import json
# import pandas as pd
import os
# import sys
# import re
import yaml

# from bokeh.layouts import column, row, layout, gridplot
# from bokeh.plotting import figure, output_file, show
# from bokeh.sampledata.autompg import autompg
# from bokeh.transform import jitter
from bokeh.palettes import Category10
from bokeh.models import HoverTool, Div
from bokeh.plotting import figure, output_file, show
# from bokeh.models import Legend
# from bokeh.models import ColumnDataSource, CategoricalTicker, Div
# from bokeh.models import ColumnDataSource, DataTable, DateFormatter, TableColumn
# from bokeh.transform import jitter
# from collections import defaultdict
from datetime import datetime as dt
from torchbenchmark.util.data import load_data_dir, load_data_files
from torchbenchmark.score.compute_score import compute_score
from torchbenchmark.score.generate_score_config import generate_bench_cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", nargs='+', 
                        help="One or more directories containing benchmark json files. " 
                             "Each directory will be plotted as a separate series. "
                             "By default, the first file in the first directory will be used"
                             " to generate a score configuration with a target of 1000,"
                             " and everything else will be relative to that.")
    parser.add_argument("--output_html", default='plot.html', help="html file to write")
    parser.add_argument("--score_heirarchy", default='torchbenchmark/score/score.yml', 
                        help="file defining score heirarchy")
    parser.add_argument("--reference_json", required=True, 
                        help="file defining score norm values, usually first json in first data_dir")
    args = parser.parse_args()
    plot_height = 800
    plot_width = 1000

    assert len(args.data_dir) > 0, "Must provide at least one data directory"
    compare_datasets = [load_data_dir(d, most_recent_files=-1) for d in args.data_dir]
    
    with open(args.reference_json) as f:
        ref_data = json.load(f)
    with open(args.score_heirarchy) as f:
        score_heirarchy = yaml.full_load(f)
    
    score_cfg = generate_bench_cfg(score_heirarchy, ref_data, target=1000)

    p = figure(plot_width=plot_width, plot_height=plot_height,
               x_axis_type='datetime')
    p.y_range.start = 0
    xs = []
    ys = []
    for d in compare_datasets:
        scores = []
        dates = []
        for i in range(len(d._json_raw)):
            data = d._json_raw[i]
            score = compute_score(score_cfg, data)
            scores.append(score)
            pytorch_ver = data['machine_info']['pytorch_version']
            date = dt.strptime(pytorch_ver[pytorch_ver.index("dev") + len("dev"):], "%Y%m%d")
            dates.append(date)
        xs.append(dates)
        ys.append(scores)

    colors = Category10[10][:len(compare_datasets)]
    basenames = map(os.path.basename, args.data_dir)

    for x, y, color in zip(xs, ys, colors):
        p.line(x, y, color=color, line_width=2, legend_label=next(basenames))

    for x, y, color in zip(xs, ys, colors):
        p.circle(x, y, color=color)

    p.legend.location = "bottom_right"

    output_file(args.output_html)
    show(p)
