import torch
import time
import gc

if __name__ == "__main__":
    # linear = torch.nn.LogSoftmax(dim=1)
    tstart = time.time()
    gc.collect()
    linear = torch.nn.Linear(2000, 1000)
    output = torch.randn(1024, 2000).type('torch.FloatTensor')
    numbers = torch.randn(100, 100).type('torch.FloatTensor')
    gc.collect()
    print("setup: " + str(time.time() - tstart))

    torch.set_num_threads(16)

    tstart = time.time()
    for _ in range(500):
#     while True:
        # output = output.sigmoid()
        # output = output.sigmoid()
        numbers = linear(output)
        # n1, n2, n3, n4 = numbers.chunk(4, 1)
        # n1 = n1.sigmoid()
        # n2 = n2.tanh()
        # n3 = n3.sigmoid()
        # n4 = n4.tanh()
        # numbers = (n1 * n2) + (n3 * n4)
        # numbers = (numbers * numbers) + (numbers * numbers)
        # numbers = numbers.transpose(0, 1)
        # numbers = numbers.tanh()
        # numbers = numbers.transpose(0, 1)
        # numbers = numbers.sigmoid()
        # numbers = numbers.transpose(0, 1)
        # numbers = numbers.tanh()
        # numbers = numbers.transpose(0, 1)
        # numbers = numbers.sigmoid()
    print("elapsed: " + str(time.time() - tstart))
