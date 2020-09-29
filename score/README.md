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
`python compute_score.py --configuration <cfg> --benchmark_data_file <data-file> `
Or, use `--benchmark_data_dir` instead, pointing to a directory containing multiple json files to compute a table of scores.

## New score versions
Periodically, as more workloads have been added to the torchbenchmark suite, or as changes to
relative weights or categories have been proposed, a new score configuration should be generated
rather than modifying an existing score definition. 

See `python generate_score_config.py -h` 

## Issues and Next Steps
For accurate score comparisons, measurements should be computed on the same machine 
(or at least same machine spec) as the data used to produce normalization constants 
in the score configuration.  

- compute_score.py should assert the machine type matches by default
- currently, a circleCI 'medium' gpu worker was used for the normalization data
- soon, a particular CPU/GPU config should be deliberately selected along with a 
  list of models/categories to be frozen for first long-living rev of the score
  
