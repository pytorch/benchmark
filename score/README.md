# Torchbench Score

Torchbench provides a normalized benchmark score similar to 'spec' or other computing benchmarks.

This is a prototype.  Current status and limitations are described below.

## Score versioning
The score is versioned, meaning only a certain set of benchmarks are captured in a particular 
version of the score (even if additional benchmarks are added to the suite).  The relative weight
of each benchmark in the overall score is frozen, along with a normalization factor measured on
a particular 'gold' machine with a particular PyTorch release.  The intent is to measure the effect
of new pytorch versions on the same workloads using the same reference machine using a consistent
benchmark configuration.

## Computing the score
To compute the current score, provide a score config and benchmark data produced by pytest with `--benchmark-json` or related arguments.
`python compute_score.py --configuration <cfg> --benchmark_data <data>`

## New score versions
Periodically, as more workloads have been added to the torchbenchmark suite, or as changes to
relative weights or categories have been proposed, a new score configuration should be generated
rather than modifying an existing score definition. 

See `python generate_score_config.py -h` 