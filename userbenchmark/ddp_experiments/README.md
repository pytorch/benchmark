This is a benchmark for benchmarking ddp (originally targeted for torchdynamo). It sweeps over:
* A set of models
* A set of numbers of nodes
* A set of torchdynamo configurations (no torchdynamo, torchdynamo + eager backend, torchdynamo + nvfuser backend, torchdynamo + inductor backend)

From these we can gather stats about DDP performance on different numbers of nodes and with different torchdynamo settings.

The script will automatically submit jobs via submitit to slurm.

An example run command. Intermediate results and startup scripts are dumped into the --job_dir FOLDER; full results can be gathered using parse_ddp.py.
```
python userbenchmark/ddp_experiments/__init__.py --partition scavenge --job_dir /fsx/users/dberard/ddp_experiments
```

TODO: merge this into the userbenchmark flow so that we can automate these tests and parse the results automatically.

An example usage of `parse_ddp.py`:
```
$ python parse_ddp.py --csv ddp_experiments_20220924-011951.csv --results_dir logs_sep23
hf_T5_large:
backend                         1_latency      2_latency  4_latency    8_latency    12_latency    16_latency    20_latency    24_latency
------------------------------  -----------  -----------  -----------  -----------  ------------  ------------  ------------  ------------
eager wo/breaks                 2525.265         6139.28  20147.442    52880.266    73171.973     Traceback     Traceback     waiting..
torchdynamo_inductor wo/breaks  1330.754         7487.04  17936.669    47601.783    81475.348     94929.492     Traceback     waiting..
eager w/breaks                  2436.703         8466.19  Traceback    Traceback    Traceback     Traceback     Traceback     waiting..
torchdynamo_inductor w/breaks   Traceback        7493.86  20017.061    55007.250    71042.156     91937.727     Traceback     waiting..

timm_vision_transformer_large:
backend                           1_latency    2_latency    4_latency    8_latency    12_latency  16_latency    20_latency    24_latency
------------------------------  -----------  -----------  -----------  -----------  ------------  ------------  ------------  ------------
eager wo/breaks                     1818.86      9976.19      29656.8      63026.9       98011.1  138281.312    Traceback     waiting..
torchdynamo_inductor wo/breaks      2151.91      9101.07      28259.4      61879.1      100458    Traceback     Traceback     waiting..
eager w/breaks                      1820.05     10648.5       29473.5      68140.4       98276.5  137554.141    Traceback     waiting..
torchdynamo_inductor w/breaks       2150.31      9409.63      28694.8      59813.4      119619    156428.422    Traceback     waiting..

hf_T5:
backend                           1_latency    2_latency    4_latency    8_latency    12_latency    16_latency  20_latency    24_latency
------------------------------  -----------  -----------  -----------  -----------  ------------  ------------  ------------  ------------
eager wo/breaks                     565.493      651.237      2105.32      4711.37       7427.44       9373.43  Traceback     waiting..
torchdynamo_inductor wo/breaks      400.017      518.402      3061.63      5166.12       8842.43      10282.8   Traceback     waiting..
eager w/breaks                      554.088     1132.43       2481.49      4734.84       5284.03       9903.49  Traceback     waiting..
torchdynamo_inductor w/breaks       400.656      526.567      2837.49      5228.22       6028.57      11748.4   Traceback     waiting..

resnet50:
backend                           1_latency    2_latency    4_latency    8_latency    12_latency    16_latency  20_latency    24_latency
------------------------------  -----------  -----------  -----------  -----------  ------------  ------------  ------------  ------------
eager wo/breaks                     132.217      188.277      248.735     2366.05        5029.29       5946.59  Traceback     waiting..
torchdynamo_inductor wo/breaks       67.041      129.034      188.435     3417.31        5716.82       5998.94  7024.278      waiting..
eager w/breaks                      118.063      202.884      293.137     2450.38        5321.45       5148.03  Traceback     waiting..
torchdynamo_inductor w/breaks        67.011      128.836      173.833      318.773       5943.41       5965.77  Traceback     waiting..
```

Run with the `--csv_out` flag to get this data in csv output for more convenient use in spreadsheets etc.

Supported options (not-exhaustive):
* `--job_dir [folder]` to select the shared job folder. It must be accessible by all the child jobs that are spawned.
* `--distributed [ddp|?]` to select the type of distributed workflow to test
* `--trainer torchbenchmark.util.distributed.core_model.trainer.Trainer` to select the trainer (e.g. can use e2e trainer)
* `--timeout {minutes}` to reduce/increase time slurm job timeout
* `--partition {train|scavenge|etc}` to choose the slurm job queue
