# How to enable TorchBench Analyzer in pytorch/benchmark?
TorchBench Analyzer has been merged to the main branch after Jun 1, 2022 .

To enable its FLOPS computation capability, you have to add
```bash
--flops dcgm
```
to the argument list when you run run.py.

The last part in the final output of pytorch/benchmark looks like the following.
```bash
GPU Time:             12.097 milliseconds
CPU Total Wall Time:  12.137 milliseconds
FLOPS:               1.9684 TFLOPs per second
```
To enable its detailed GPU metric records export capability, you have to add
```bash
--flops dcgm --export-dcgm-metrics
```
to the argument list when you run run.py. The metrics will be output to [model_name]_all_metrics.csv automatically.


# How to integrate TorchBench Analyzer into your project?
The following code snippet shows the basic usage of TorchBench Analyzer. TorchBench Analyzer is integrated into pytorch/benchmark. If you want to use it to test your own function, the only thing you need to do is copy the whole folder pytorch/benchmark/components/model_analyzer to your project and code like the following.

```python
from model_analyzer.TorchBenchAnalyzer import ModelAnalyzer

def work():
    # A simple mm test 
    import torch
    n=4096
    x = torch.ones((n, n), dtype=torch.float32, device="cuda")
    y = torch.ones((n, n),dtype=torch.float32, device="cuda")

    # configure model analyzer
    model_analyzer = ModelAnalyzer()
    model_analyzer.start_monitor()

    # run your own code
    for i in range(200):
        if i % 100 == 0:
            print(i)
        torch.mm(x, y)
    
    # stop and aggregate the profiling results
    model_analyzer.stop_monitor()
    model_analyzer.aggregate()
    tflops = model_analyzer.calculate_flops()
    print('{:<20} {:>20}'.format("FLOPS:", "%.4f TFLOPs per second" % tflops, sep=''))
```

# Different GPU metrics collection backend support

TorchBench Analyzer supports different GPU metrics collection backend, dcgm, nvml, and fvcore. The backend priority is dcgm > nvml > fvcore. Currently, the metrics exposed to TorchBench are [cpu_peak_mem,gpu_peak_mem,flops]. The following table shows what kind of metrics are supported by different backends.

| Backend | gpu_peak_mem | flops |
| ------- | ------------ | ----- |
| dcgm    | Yes          | Yes   |
| nvml    | Yes          | No    |
| fvcore  | No           | Yes   |

cpu_peak_mem is always collcted by psutil.Process. More metrics will be enabled in the future.

For TorchBench, we use nvml to collect gpu_peak_mem, fvcore to collect flops, and psutil.Process to collect cpu_peak_mem by default. We didn't add fvcore to backend list. If you want to use fvcore to collect flops, you can add flop metric to metric list but ignore the backend option.

