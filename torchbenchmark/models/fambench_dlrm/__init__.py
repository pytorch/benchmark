"""
Simplifed dlrm model from FAMBench
It doesn't support multiGPU or fbgemm_gpu.
"""
import torch
import sys
import os
import numpy as np
import torch.nn as nn
from torchbenchmark import REPO_PATH
from typing import Tuple

# Import FAMBench model path
class add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass
DLRM_PATH = os.path.join(REPO_PATH, "submodules", "FAMBench", "benchmarks", "dlrm", "ootb")
with add_path(DLRM_PATH):
    from dlrm_s_pytorch import DLRM_Net
    from dlrm_data_pytorch import make_random_data_and_loader
    import optim.rwsadagrad as RowWiseSparseAdagrad

from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.tasks import RECOMMENDATION
from .config import FAMBenchTrainConfig, FAMBenchEvalConfig, cfg_to_str
from .args import parse_fambench_args, validate_fambench_args
from .data import prep_data

class Model(BenchmarkModel):
    task = RECOMMENDATION.RECOMMENDATION
    FAMBENCH_MODEL = True
    # config
    DEFAULT_EVAL_ARGS = FAMBenchEvalConfig()
    DEFAULT_TRAIN_ARGS = FAMBenchTrainConfig()
    DEFAULT_EVAL_BATCH_SIZE = DEFAULT_EVAL_ARGS.mini_batch_size
    DEFAULT_TRAIN_BATCH_SIZE = DEFAULT_TRAIN_ARGS.mini_batch_size
    # run only 1 batch
    DEFAULT_NUM_BATCHES = 1

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(self, test, device, batch_size, jit, extra_args)
        if test == "train":
            self.fambench_args = parse_fambench_args(cfg_to_str(self.DEFAULT_TRAIN_ARGS))
        elif test == "eval":
            self.fambench_args = parse_fambench_args(cfg_to_str(self.DEFAULT_EVAL_ARGS))
            self.fambench_args.inference_only = True
        if device == "cuda":
            self.fambench_args.use_gpu = True
        self.fambench_args.num_batches = self.DEFAULT_NUM_BATCHES
        args = self.fambench_args
        validate_fambench_args(args)
        self.prep(args)
        prep_data(args)
        # in TorchBench test, we only use 1 GPU
        ndevices = 1
        dlrm = DLRM_Net(
            m_spa,
            ln_emb,
            ln_bot,
            ln_top,
            args.arch_project_size,
            arch_interaction_op=args.arch_interaction_op,
            arch_interaction_itself=args.arch_interaction_itself,
            sigmoid_bot=-1,
            sigmoid_top=ln_top.size - 2,
            sync_dense_params=args.sync_dense_params,
            loss_threshold=args.loss_threshold,
            ndevices=ndevices,
            qr_flag=args.qr_flag,
            qr_operation=args.qr_operation,
            qr_collisions=args.qr_collisions,
            qr_threshold=args.qr_threshold,
            md_flag=args.md_flag,
            md_threshold=args.md_threshold,
            weighted_pooling=args.weighted_pooling,
            loss_function=args.loss_function,
            learning_rate=args.learning_rate,
            use_gpu=args.use_gpu,
            use_fbgemm_gpu=args.se_fbgemm_gpu,
            fbgemm_gpu_codegen_pref=args.fbgemm_gpu_codegen_pref,
            inference_only=args.inference_only,
            quantize_mlp_with_bit=args.quantize_mlp_with_bit,
            quantize_emb_with_bit=args.quantize_emb_with_bit,
            use_torch2trt_for_mlp=args.use_torch2trt_for_mlp,)
        # In dlrm.quantize_embedding called below, the torch quantize calls run
        # on cpu tensors only. They cannot quantize tensors stored on the gpu.
        # So quantization occurs on cpu tensors before transferring them to gpu if
        # use_gpu is enabled.
        if args.quantize_emb_with_bit != 32:
            dlrm.quantize_embedding(args.quantize_emb_with_bit)
        if not args.inference_only:
            assert args.quantize_mlp_with_bit == 32, (
                "Dynamic quantization for mlp requires "
                + "--inference-only because training is not supported"
            )
        else:
            # Currently only INT8 and FP16 quantized types are supported for quantized MLP inference.
            # By default we don't do the quantization: quantize_{mlp,emb}_with_bit == 32 (FP32)
            assert args.quantize_mlp_with_bit in [
                8,
                16,
                32,
            ], "only support 8/16/32-bit but got {}".format(args.quantize_mlp_with_bit)

            if not args.use_torch2trt_for_mlp:
                if args.quantize_mlp_with_bit == 16 and args.use_gpu:
                    dlrm.top_l = dlrm.top_l.half()
                    dlrm.bot_l = dlrm.bot_l.half()
                elif args.quantize_mlp_with_bit in [8, 16]:
                    assert not args.use_gpu, (
                        "Cannot run PyTorch's built-in dynamic quantization for mlp "
                        + "with --use-gpu enabled, because DynamicQuantizedLinear's "
                        + "forward function calls 'quantized::linear_dynamic', which does not "
                        + "support the 'CUDA' backend. To convert to and run quantized mlp layers "
                        + "on the gpu, install torch2trt and enable --use-torch2trt-for-mlp. "
                        + "Alternatively, disable --use-gpu to use PyTorch's built-in "
                        + "cpu quantization ops for the mlp layers. "
                    )
                    if args.quantize_mlp_with_bit == 8:
                        quantize_dtype = torch.qint8
                    else:
                        quantize_dtype = torch.float16
                    dlrm.top_l = torch.quantization.quantize_dynamic(
                        dlrm.top_l, {torch.nn.Linear}, quantize_dtype
                    )
                    dlrm.bot_l = torch.quantization.quantize_dynamic(
                        dlrm.bot_l, {torch.nn.Linear}, quantize_dtype
                    )
            # Prep work for embedding tables and model transfer:
            # Handling single-cpu and single-gpu modes
            # NOTE: This also handles dist-backend modes (CLI args --dist-backend=nccl,
            # --dist-backend=ccl, and --dist-backend=mpi) because in these modes each
            # process runs in single-gpu mode. For example, if 8 processes are launched
            # running dlrm_s_pytorch.py with --dist-backend=nccl --use-gpu, each process
            # will run in single-gpu mode, resulting in 8 gpus total running distributed
            # training or distributed inference if --inference-only is enabled.
            if dlrm.ndevices_available <= 1:
                assert not args.use_fbgemm_gpu, "fbgemm_gpu is not supported."
                if args.use_gpu:
                    dlrm = dlrm.to(device)
                    if dlrm.weighted_pooling == "fixed":
                        for k, w in enumerate(dlrm.v_W_l):
                            dlrm.v_W_l[k] = w.to(device)
            else:
                # Handing Multi-gpu mode
                dlrm.bot_l = dlrm.bot_l.to(device)
                dlrm.top_l = dlrm.top_l.to(device)
                dlrm.prepare_parallel_model(ndevices)
        assert not args.use_torch2trt_for_mlp, "torch2trt is not supported."
        if not args.inference_only:
            # specify the optimizer algorithm
            opts = {
                "sgd": torch.optim.SGD,
                "rwsadagrad": RowWiseSparseAdagrad.RWSAdagrad,
                "adagrad": torch.optim.Adagrad,
            }
            assert ext_dist.my_size == 1, "Distributed is not supported."
            parameters = (
                dlrm.parameters()
                if ext_dist.my_size == 1
                else [
                    {
                        "params": [
                            p
                            for emb in (
                                [e.fbgemm_gpu_emb_bag for e in dlrm.fbgemm_emb_l]
                                if use_fbgemm_gpu
                                else dlrm.emb_l_q
                                if dlrm.quantize_bits != 32
                                else dlrm.emb_l
                            )
                            for p in emb.parameters()
                        ],
                        "lr": args.learning_rate,
                    },
                    # TODO check this lr setup
                    # bottom mlp has no data parallelism
                    # need to check how do we deal with top mlp
                    {
                        "params": dlrm.bot_l.parameters(),
                        "lr": args.learning_rate,
                    },
                    {
                        "params": dlrm.top_l.parameters(),
                        "lr": args.learning_rate,
                    },
                ]
            )
            self.optimizer = opts[args.optimizer](parameters, lr=args.learning_rate)
            self.lr_scheduler = LRPolicyScheduler(
                optimizer,
                args.lr_num_warmup_steps,
                args.lr_decay_start_step,
                args.lr_num_decay_steps,
            )
        # Guarantee GPU setup has completed before training or inference starts.
        if args.use_gpu:
            torch.cuda.synchronize()

    def prep(self, args):
        np.random.seed(args.numpy_rand_seed)
        np.set_printoptions(precision=args.print_precision)
        torch.set_printoptions(args.print_precision)
        torch.manual_seed(args.numpy_rand_seed)
        if args.test_mini_batch_size < 0:
            # if the parameter is not set, use the training batch size
            args.test_mini_batch_size = args.mini_batch_size
        if args.test_num_workers < 0:
            # if the parameter is not set, use the same parameter for training
            args.test_num_workers = args.num_workers
        if args.use_gpu:
            torch.cuda.manual_seed_all(args.numpy_rand_seed)
            torch.backends.cudnn.deterministic = True