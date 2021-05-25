import dataclasses

@dataclasses.dataclass
class SACConfig:
    env_id = "Pendulum-v0"
    seed = 123
    num_steps = 1
    transitions_per_step = 1
    max_episode_steps = 10
    batch_size = 512
    tau = 0.005
    actor_lr = 1e-4
    critic_lr = 1e-4
    gamma = 0.99
    init_alpha = 0.1
    alpha_lr = 1e-4
    buffer_size = 1_000_000
    eval_interval = 5000
    eval_episodes = 10
    warmup_steps = 1
    render = False
    actor_clip = 0.0
    critic_clip = 0.0
    name = "sac_run"
    actor_l2 = 0.0
    critic_l2 = 0.0
    target_delay = 2
    actor_delay = 1
    save_interval = 100_000
    verbosy = 0
    gradient_updates_per_step = 1
    prioritized_replay = False
    skip_save_to_disk = True
    skip_log_to_disk = True
    discrete_actions = True
    log_std_low = -10.0
    log_std_high = 2.0
    self_regularized = False
    sr_max_critic_updates_per_step = 10
    sr_critic_target_improvement_init = 0.7
    sr_critic_target_improvement_final = 0.9
    
