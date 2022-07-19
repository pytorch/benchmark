"""
This is a test file for TorchBenchAnalyzer
"""

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
    # run the computation part
    for i in range(200):
        if i % 100 == 0:
            print(i)
        torch.mm(x, y)
    # start test app here
    # run_app(4096)
    
    # stop and aggregate the profiling results
    model_analyzer.stop_monitor()
    model_analyzer.aggregate()
    tflops = model_analyzer.calculate_flops()
    print('{:<20} {:>20}'.format("FLOPS:", "%.4f TFLOPs per second" % tflops, sep=''))


if __name__ == "__main__":
    work()
