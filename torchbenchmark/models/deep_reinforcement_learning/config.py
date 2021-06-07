import torch
import dataclasses

@dataclasses.dataclass
class DQNConfig():
    target_update = 2000
    T_max = 100000
    learn_start = 2
    memory_capacity = 100000
    replay_frequency = 4
    multi_step = 3
    seed = 123
    max_episode_length = int(108e3)
    atoms = 51
    norm_clip = 10
    history_length = 4
    atmos = 51
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
    priority_exponent = 0.5
    priority_weight = 0.4
    game = "tetris"
    model = None
    def __init__(self, device=None):
        self.device = torch.device(device)
    
