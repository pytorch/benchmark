# RNN benchmarks

To run all the benchmarks, and get a summary view, use `python runner.py`

To run a specific benchmark, call it as a Python module:
`python -m benchmarks.sru`. They come with a lot of command line options
for fine-tuning.

## Caveats

Use Linux for the most accurate timing. A lot of these tests only run
on CUDA.
