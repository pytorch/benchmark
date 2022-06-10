import os
import datetime
import time
import json
from pathlib import Path

def dump_output(bm_name, output):
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    target_dir = current_dir.parent.joinpath(".userbenchmark", bm_name)
    target_dir.mkdir(exist_ok=True, parents=True)
    fname = "metrics-{}.json".format(datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S"))
    full_fname = os.path.join(target_dir, fname)
    with open(full_fname, 'w') as f:
        json.dump(output, f, indent=4)
