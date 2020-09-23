from bokeh.layouts import column
from bokeh.models import ColumnDataSource, CategoricalTicker
from bokeh.plotting import figure, output_file, show
from bokeh.sampledata.autompg import autompg
from bokeh.transform import jitter

import argparse
import json
import pandas as pd
import sys


# def load_json(filename):
#     with open(filename) as f:
#         data = json.load(f)
#     return data
# te_raw = load_json('te.json')
# old_raw = load_json('old.json')

# te_data = te_raw['benchmarks'][0]['stats']['data']
# old_data = old_raw['benchmarks'][0]['stats']['data']
# data = {
#     'fuser' : ['te'] * len(te_data) + ['old'] * len(old_data),
#     'time': te_data + old_data
# }
# datasrc = ColumnDataSource(data)
# p1 = figure(x_range=['old', 'te'], title="Transformer: Old vs TE fuser")
# p1.xgrid.grid_line_color = None
# p1.circle(x=data['fuser'], y=data['time'])

# p2 = figure(x_range=['old', 'te'], title="Transformer: Old vs TE fuser")
# p2.xgrid.grid_line_color = None
# p2.circle(x='fuser', y='time', source=datasrc)
# import ipdb; ipdb.set_trace()
# output_file("te_old.html")
# show(column(p1, p2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare_files", nargs='+', type=argparse.FileType('r'),
                        help='json data files formatted by pytest-benchmark module')
    parser.add_argument("--benchmark_names", nargs='+',
                        help="names of benchmarks expected to exist in each of the"
                             " provided json files")
    parser.add_argument("--output_html", default='plot.html', help="html file to write")
    args = parser.parse_args()
    if args.compare_files is None or len(args.compare_files) == 0:
        print("Must provide at least one json file")
        sys.exit(-1)
    
    if args.benchmark_names is None or len(args.benchmark_names) == 0:
        json_data = json.load(args.compare_files[0])
        names = [json_data['benchmarks'][i]['name'] for i in range(len(json_data['benchmarks']))]
        print("Please specify list of benchmark names to plot."
              " Found these benchmarks in provided json:")
        for name in names:
            print(name)
        sys.exit(-1)
    
    data = {name: pd.DataFrame() for name in args.benchmark_names}
    tags = [f.name for f in args.compare_files]
    for f, tag in zip(args.compare_files, tags):
        json_data = json.load(f)
        for benchmark in json_data['benchmarks']:
            if benchmark['name'] in data:
                new_df = pd.DataFrame()  \
                    .assign(time=benchmark['stats']['data']) \
                    .assign(tag=tag) \
                    .assign(commit=json_data['commit_info']['id']) \
                    .assign(date=json_data['commit_info']['time'])
                data[benchmark['name']] = data[benchmark['name']].append(new_df,
                                                                         ignore_index=True)

    plots = []
    for name in args.benchmark_names:
        p = figure(x_range=tags, title=name)
        p.xgrid.grid_line_color = None
        # p.xaxis.major_label_orientation = "vertitcal"
        p.circle(x='tag', y='time', source=data[name])
        plots.append(p)
    output_file(args.output_html)
    show(column(*plots))


    