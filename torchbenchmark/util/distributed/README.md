## Distributed Training Paradigms

This folder contains native distributed training paradigms from PyTorch. The
current structure makes distributed training paradigms orthogonal to models.
Users can specify model and trainer when launching experiments.


```
$ python -m torchbenchmark.util.distributed.submit -h
usage: submit.py [--ngpus NGPUS] [--nodes NODES] [--timeout TIMEOUT] [--profiler PROFILER] [--partition PARTITION] [--job_dir JOB_DIR] [--model MODEL]
                 [--trainer TRAINER]

Submitit for PyTorch Distributed Benchmark

optional arguments:
  --ngpus NGPUS         Number of gpus to request on each node
  --nodes NODES         Number of nodes to request
  --timeout TIMEOUT     Duration of the job
  --profiler PROFILER   Measure with PyTorch Profiler. Disabled by default, as it crashes on AWS
  --partition PARTITION
                        The Slurm partition to submit to
  --job_dir JOB_DIR     A shared folder across all worker processes
  --model MODEL         specify the model to experiment with, by default uses e2e_models.hf_bert
  --trainer TRAINER     training paradigm, by default using DDP
```

### Launch Single-Node Experiments

If `--ngpus` and `--nodes` are not specified, the script will submit a job to
Slurm that uses 1 node and 2 GPUs. The first two lines of the outputs are the
rendezvous file path and the Slurm job id. When job is done, it will print the
measurements of forward, backward, and optimizer step across all ranks.
```
$ python -m torchbenchmark.util.distributed.submit --partition=train --job_dir=/shared/shenli/logs/
/shared/shenli/logs/b0b94763a518461f80d23831fbf30e45_init
43513
[{'fwd_mean': 13.047599983215331, 'fwd_stdev': 0.09282855543720885, 'bwd_mean': 27.015679931640626, 'bwd_stdev': 6.021353625347652, 'opt_mean': 12.986438274383545, 'opt_stdev': 0.18885086070407958}, {'fwd_mean': 12.769235229492187, 'fwd_stdev': 0.0986564824823847, 'bwd_mean': 27.083469200134278, 'bwd_stdev': 6.100107700108071, 'opt_mean': 13.155091094970704, 'opt_stdev': 0.1362884411184391}]
```

### Launch Multi-Node Experiments

The command below submits a 16-GPU (8 GPU per node) experiment to the train
partition. Note that submitit will use the provided `--job_dir` to create a file
for rendezvous. During the rendezvous, all ranks will try to read and write from
the file. Therefore, for large scale experiment, please choose a fast NFS as the
`--job_dir` to avoid long rendezvous delay.

```
$ python -m torchbenchmark.util.distributed.submit --partition=train --job_dir=/shared/shenli/logs/ --nodes=2 --ngpus=8
/data/home/shenli/projects/tmp/0cfe1d32536e4bd994ff6844a2d47e03_init
43503
[{'fwd_mean': 14.954096031188964, 'fwd_stdev': 0.14958178449663892, 'bwd_mean': 175.00098419189453, 'bwd_stdev': 83.48690790640256, 'opt_mean': 14.680272006988526, 'opt_stdev': 0.25537006136939977}, {'fwd_mean': 15.037401580810547, 'fwd_stdev': 0.36717764480028603, 'bwd_mean': 175.1401466369629, 'bwd_stdev': 83.4840124424261, 'opt_mean': 14.401081657409668, 'opt_stdev': 0.2478409136738749}, {'fwd_mean': 16.111654376983644, 'fwd_stdev': 1.204402844824517, 'bwd_mean': 172.26629104614258, 'bwd_stdev': 82.85572057742398, 'opt_mean': 16.210534191131593, 'opt_stdev': 1.2796832893260388}, {'fwd_mean': 15.346086311340333, 'fwd_stdev': 0.19084059805242357, 'bwd_mean': 173.8418182373047, 'bwd_stdev': 83.42179411126631, 'opt_mean': 15.47025260925293, 'opt_stdev': 0.3551752132621945}, {'fwd_mean': 15.604217624664306, 'fwd_stdev': 0.12541896985471224, 'bwd_mean': 174.2532615661621, 'bwd_stdev': 83.38071585042276, 'opt_mean': 14.743187141418456, 'opt_stdev': 0.07535304310108071}, {'fwd_mean': 15.332918453216553, 'fwd_stdev': 0.8900351088440596, 'bwd_mean': 174.83438110351562, 'bwd_stdev': 83.46463720110432, 'opt_mean': 14.410524845123291, 'opt_stdev': 0.17680403063548852}, {'fwd_mean': 16.149644660949708, 'fwd_stdev': 1.5805064990072413, 'bwd_mean': 172.51102676391602, 'bwd_stdev': 83.61162884055666, 'opt_mean': 15.899552154541016, 'opt_stdev': 0.36785963810918}, {'fwd_mean': 15.462988758087159, 'fwd_stdev': 0.461404784497284, 'bwd_mean': 172.78116149902343, 'bwd_stdev': 83.80683844756977, 'opt_mean': 16.229638481140135, 'opt_stdev': 1.2433197174143409}, {'fwd_mean': 15.314476680755615, 'fwd_stdev': 0.4955161979972898, 'bwd_mean': 174.39610900878907, 'bwd_stdev': 84.05470965496554, 'opt_mean': 14.703580856323242, 'opt_stdev': 0.35581850306214335}, {'fwd_mean': 15.119887828826904, 'fwd_stdev': 0.4269950332877091, 'bwd_mean': 174.61544952392578, 'bwd_stdev': 83.48891819434135, 'opt_mean': 14.694771099090577, 'opt_stdev': 0.25967125498698657}, {'fwd_mean': 15.140227222442627, 'fwd_stdev': 0.1364234546230897, 'bwd_mean': 173.63630142211915, 'bwd_stdev': 83.7641599477111, 'opt_mean': 15.64806079864502, 'opt_stdev': 0.16184027390324782}, {'fwd_mean': 15.480313396453857, 'fwd_stdev': 0.6563934904441636, 'bwd_mean': 173.14631881713868, 'bwd_stdev': 83.94256111977732, 'opt_mean': 15.796409797668456, 'opt_stdev': 0.3478268126614144}, {'fwd_mean': 15.405878353118897, 'fwd_stdev': 0.12069590977857436, 'bwd_mean': 173.14385986328125, 'bwd_stdev': 83.72500444753977, 'opt_mean': 15.883203125, 'opt_stdev': 0.09908576862460719}, {'fwd_mean': 14.758016014099121, 'fwd_stdev': 0.2400985187825859, 'bwd_mean': 175.51165618896485, 'bwd_stdev': 83.89509049709447, 'opt_mean': 14.171260643005372, 'opt_stdev': 0.07589076767179384}, {'fwd_mean':
14.958608055114746, 'fwd_stdev': 0.425216973486159, 'bwd_mean': 174.43686141967774, 'bwd_stdev': 83.61664420857524, 'opt_mean': 14.976755332946777, 'opt_stdev': 0.9215972074088493}, {'fwd_mean': 15.412767791748047, 'fwd_stdev': 0.43232722408417457, 'bwd_mean': 173.27083435058594, 'bwd_stdev': 83.66419263725956, 'opt_mean': 15.738156700134278, 'opt_stdev': 0.352595499170623}]
```

### Files

* `submit.py`: a helper that submits Slurm jobs.
* `trainer.py`: the abstract class of all distributed training paradigms.
* `ddp.py`: a subclass of Trainer that implements DDP training.