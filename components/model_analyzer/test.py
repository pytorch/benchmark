"""
"""

from TorchBenchAnalyzer import ModelAnalyzer

def work():
    # A simple test 
    import torch
    from torch.utils.benchmark import Timer
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
    model_analyzer.print_flops()


if __name__ == "__main__":
    work()
