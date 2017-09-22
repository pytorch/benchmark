import torch


class TimeBroadcast:
    def setup(self):
        self.d = torch.zeros(1000, 1000)
        self.e = torch.zeros(1)

    def time_broadcast(self):
        for i in range(500):
            self.d * self.e

    def time_no_broadcast(self):
        for i in range(500):
            self.d * self.d
