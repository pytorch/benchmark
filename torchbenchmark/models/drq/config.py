import dataclasses

@dataclasses.dataclass
class DRQConfig:
    env = "cartpole_swingup"
    # IMPORTANT: if action_repeat is used the effective number of env steps needs to be
    # multiplied by action_repeat in the result graphs.
    # This is a common practice for a fair comparison.
    # See the 2nd paragraph in Appendix C of SLAC: https://arxiv.org/pdf/1907.00953.pdf
    # See Dreamer TF2's implementation: https://github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c/dreamer.py#L340
    action_repeat = 4
    # train
    num_train_steps = 1
    num_train_iters = 1
    num_seed_steps = 1000
    replay_buffer_capacity = 100000
    seed = 1
    # eval
    eval_frequency = 5000
    # observation
    image_size = 84
    image_pad = 4
    frame_stack = 3
    # global params
    lr = 1e-3
    # IMPORTANT: please use a batch size of 512 to reproduce the results in the paper. Hovewer, with a smaller batch size it still works well.
    batch_size = 128
