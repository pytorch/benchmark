import torch
import dataclasses

@dataclasses.dataclass
class DQNConfig():
    target_update = 2000
    T_max = 100000
    learn_start = 1600
    memory_capacity = 100000
    replay_frequency = 1
    multi_step = 20
    architecture = "data-efficient"
    hidden_size = 256
    learning_rate = 0.0001
    evaluation_interval = 10000
    noisy_std = 0.1
    V_min = -10
    V_max = 10
    batch_size = 32
    evaluate = False
    enable_cudnn = True
    discount = 0.99
    reward_clip = 1
    adam_eps = 1.5e-4
    game = "tetris"
    def __init__(self, device=None):
        self.device = torch.device(device)
    
