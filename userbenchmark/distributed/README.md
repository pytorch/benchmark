This is a benchmark for measuring PyTorch Distributed performance. 

An example run command. Results are outputted as a json file in --job_dir FOLDER.
```
python run_benchmark.py distributed --ngpus 8 --nodes 1  --model torchbenchmark.e2e_models.hf_bert.Model --trainer torchbenchmark.util.distributed.trainer.Trainer --distributed ddp --job_dir $PWD/.userbenchmark/distributed/e2e_hf_bert --profiler False
```
Supported options (not-exhaustive):
* --distributed {torchbenchmark.e2e_models.hf_bert.Model, torchbenchmark.e2e_models.hf_t5.Model}
* --profiler {True, False}
  * Returns one trace for every GPU, saved into --job_dir FOLDER.  
* --distributed {ddp, fsdp, deepspeed}
