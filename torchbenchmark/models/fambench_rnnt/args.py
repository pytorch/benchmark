import argparse

def get_train_args(args):
    parser = argparse.ArgumentParser(description='RNN-T Training Reference')

    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', default=100, type=int,
                          help='Number of epochs for the entire training')
    training.add_argument("--warmup_epochs", default=6, type=int,
                          help='Initial epochs of increasing learning rate')
    training.add_argument("--hold_epochs", default=40, type=int,
                          help='Constant max learning rate epochs after warmup')
    training.add_argument('--epochs_this_job', default=0, type=int,
                          help=('Run for a number of epochs with no effect on the lr schedule.'
                                'Useful for re-starting the training.'))
    training.add_argument('--cudnn_benchmark', action='store_true', default=True,
                          help='Enable cudnn benchmark')
    training.add_argument('--amp', '--fp16', action='store_true', default=False,
                          help='Use mixed precision training')
    training.add_argument('--seed', default=None, type=int, help='Random seed')
    training.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int,
                          help='GPU id used for distributed training')
    training.add_argument('--target', default=0.058, type=float, help='Target WER accuracy')
    training.add_argument('--weights_init_scale', default=0.5, type=float, help='If set, overwrites value in config.')
    training.add_argument('--hidden_hidden_bias_scale', type=float, help='If set, overwrites value in config.')

    optim = parser.add_argument_group('optimization setup')
    optim.add_argument('--batch_size', default=128, type=int,
                       help='Effective batch size per GPU (might require grad accumulation')
    optim.add_argument('--val_batch_size', default=2, type=int,
                       help='Evalution time batch size')
    optim.add_argument('--lr', default=4e-3, type=float,
                       help='Peak learning rate')
    optim.add_argument("--min_lr", default=1e-5, type=float,
                       help='minimum learning rate')
    optim.add_argument("--lr_exp_gamma", default=0.935, type=float,
                       help='gamma factor for exponential lr scheduler')
    optim.add_argument('--weight_decay', default=1e-3, type=float,
                       help='Weight decay for the optimizer')
    optim.add_argument('--grad_accumulation_steps', default=8, type=int,
                       help='Number of accumulation steps')
    optim.add_argument('--log_norm', action='store_true',
                       help='If enabled, gradient norms will be logged')
    optim.add_argument('--clip_norm', default=1, type=float,
                       help='If provided, gradients will be clipped above this norm')
    optim.add_argument('--beta1', default=0.9, type=float, help='Beta 1 for optimizer')
    optim.add_argument('--beta2', default=0.999, type=float, help='Beta 2 for optimizer')
    optim.add_argument('--ema', type=float, default=0.999,
                       help='Discount factor for exp averaging of model weights')

    io = parser.add_argument_group('feature and checkpointing setup')
    io.add_argument('--dali_device', type=str, choices=['cpu', 'gpu'],
                    default='cpu', help='Use DALI pipeline for fast data processing')
    io.add_argument('--resume', action='store_true',
                    help='Try to resume from last saved checkpoint.')
    io.add_argument('--ckpt', default=None, type=str,
                    help='Path to a checkpoint for resuming training')
    io.add_argument('--save_at_the_end', action='store_true',
                    help='Saves model checkpoint at the end of training')
    io.add_argument('--save_frequency', default=None, type=int,
                    help='Checkpoint saving frequency in epochs')
    io.add_argument('--keep_milestones', default=[], type=int, nargs='+',
                    help='Milestone checkpoints to keep from removing')
    io.add_argument('--save_best_from', default=200, type=int,
                    help='Epoch on which to begin tracking best checkpoint (dev WER)')
    io.add_argument('--val_frequency', default=1, type=int,
                    help='Number of epochs between evaluations on dev set')
    io.add_argument('--log_frequency', default=25, type=int,
                    help='Number of steps between printing training stats')
    io.add_argument('--prediction_frequency', default=None, type=int,
                    help='Number of steps between printing sample decodings')
    io.add_argument('--model_config', default='configs/baseline_v3-1023sp.yaml',
                    type=str, required=True,
                    help='Path of the model configuration file')
    io.add_argument('--num_buckets', type=int, default=6,
                    help='If provided, samples will be grouped by audio duration, '
                         'to this number of backets, for each bucket, '
                         'random samples are batched, and finally '
                         'all batches are randomly shuffled')
    io.add_argument('--train_manifests', type=str, required=True, nargs='+',
                    help='Paths of the training dataset manifest file')
    io.add_argument('--val_manifests', type=str, required=True, nargs='+',
                    help='Paths of the evaluation datasets manifest files')
    io.add_argument('--max_duration', type=float,
                    help='Discard samples longer than max_duration')
    io.add_argument('--dataset_dir', required=True, type=str,
                    help='Root dir of dataset')
    # io.add_argument('--output_dir', type=str, required=True,
    #                 help='Directory for logs and checkpoints')
    # io.add_argument('--log_file', type=str, default=None,
    #                 help='Path to save the training logfile.')
    io.add_argument('--max_symbol_per_sample', type=int, default=None,
                    help='maximum number of symbols per sample can have during eval')
    # io.add_argument('--mlperf', action='store_true', help='Enable MLPerf Logging.')

    # # FB5 Logging
    # io.add_argument("--fb5logger", type=str, default=None)
    # io.add_argument("--fb5config", type=str, default="small")
    args = parser.parse_args(args)
    return args

def get_eval_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["pytorch"], default="pytorch", help="Backend")
    parser.add_argument("--scenario", choices=["SingleStream", "Offline", "Server"], default="Offline", help="Scenario")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    # parser.add_argument("--mlperf_conf", default=str(MLPERF_CONF), help="mlperf rules config")
    parser.add_argument("--user_conf", default="user.conf", help="user config for user LoadGen settings such as target QPS")
    parser.add_argument("--pytorch_config_toml", default="pytorch/configs/rnnt.toml")
    # parser.add_argument("--pytorch_checkpoint", default="pytorch/work_dir/rnnt.pt")
    # Checkpoint is not used in torchbench
    parser.add_argument("--pytorch_checkpoint", default=None)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--perf_count", type=int, default=None)
    # parser.add_argument("--log_dir", required=True)
    # # FB5 Logging
    # parser.add_argument("--fb5logger", type=str, default=None)
    # parser.add_argument("--fb5config", type=str, default="small")
    args = parser.parse_args(args)
    return args