
TorchBench CI has detected a performance signal or runtime regression.

Control PyTorch commit: 845544797242fa72ddd1bff729c71bcb5c9a9381
Control PyTorch version: 2.3.0a0+git8455447

Treatment PyTorch commit: 7aff92c838af27e3c7321d973e4b6da0b7f25ba4
Treatment PyTorch version: 2.3.0a0+git7aff92c

Affected Tests:
- test_eval[hf_Longformer-cuda-eager]_latency: -22.09732%


Tests that were no longer run on treatment commit:


Tests that were newly added on treatment commit:


Runtime regressions found?
No runtime errors were found in the new benchmarks run--you are all good there!

GitHub workflow that triggered this issue: No URL found, please look for the failing action in https://github.com/pytorch/benchmark/actions

cc @xuzhao9
