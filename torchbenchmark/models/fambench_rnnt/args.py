import argparse

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