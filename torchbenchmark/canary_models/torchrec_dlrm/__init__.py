import torch

# OSS import
try:
    # pyre-ignore[21]
    # @manual=//ai_codesign/benchmarks/dlrm/torchrec_dlrm/data:dlrm_dataloader
    from .data.dlrm_dataloader import get_dataloader
except ImportError:
    pass

import itertools
import os

from pyre_extensions import none_throws
from torch import distributed as dist
from torchbenchmark.tasks import RECOMMENDATION
from torchrec import EmbeddingBagCollection
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.shard import shard_modules
from torchrec.models.dlrm import DLRM_DCN, DLRMTrain
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter

from ...util.model import BenchmarkModel

from .args import InteractionType, parse_args


class Model(BenchmarkModel):
    task = RECOMMENDATION.RECOMMENDATION
    DEFAULT_TRAIN_BSIZE = 1024
    DEFAULT_EVAL_BSIZE = 1024
    CANNOT_SET_CUSTOM_OPTIMIZER = True
    # Deepcopy will OOM in correctness testing
    DEEPCOPY = False

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(
            test=test, device=device, batch_size=batch_size, extra_args=extra_args
        )
        args = parse_args(self.extra_args)
        backend = "nccl" if self.device == "cuda" else "gloo"
        device = torch.device(self.device)
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        if not dist.is_initialized():
            dist.init_process_group(backend=backend)

        # initialize example data
        if self.test == "train":
            args.batch_size = self.batch_size
            loader = get_dataloader(args, backend, "train")
        if self.test == "eval":
            args.test_batch_size = self.batch_size
            loader = get_dataloader(args, backend, "test")
        self.iterator = itertools.cycle(iter(loader))
        self.example_inputs = next(self.iterator).to(device)
        # parse the args
        args.dense_arch_layer_sizes = [
            int(x)
            for x in args.dense_arch_layer_sizes.split(",")
            if x.strip().isdigit()
        ]
        args.over_arch_layer_sizes = [
            int(x) for x in args.over_arch_layer_sizes.split(",") if x.strip().isdigit()
        ]
        args.interaction_branch1_layer_sizes = [
            int(x)
            for x in args.interaction_branch1_layer_sizes.split(",")
            if x.strip().isdigit()
        ]
        args.interaction_branch2_layer_sizes = [
            int(x)
            for x in args.interaction_branch2_layer_sizes.split(",")
            if x.strip().isdigit()
        ]

        assert (
            args.in_memory_binary_criteo_path == None
            and args.synthetic_multi_hot_criteo_path == None
        ), f"Torchbench only supports random data inputs."

        eb_configs = [
            EmbeddingBagConfig(
                name=f"t_{feature_name}",
                embedding_dim=args.embedding_dim,
                num_embeddings=(
                    none_throws(args.num_embeddings_per_feature)[feature_idx]
                    if args.num_embeddings is None
                    else args.num_embeddings
                ),
                feature_names=[feature_name],
            )
            for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
        ]
        dlrm_model = DLRM_DCN(
            embedding_bag_collection=EmbeddingBagCollection(
                tables=eb_configs, device=device
            ),
            dense_in_features=len(DEFAULT_INT_NAMES),
            dense_arch_layer_sizes=args.dense_arch_layer_sizes,
            over_arch_layer_sizes=args.over_arch_layer_sizes,
            dcn_num_layers=args.dcn_num_layers,
            dcn_low_rank_dim=args.dcn_low_rank_dim,
            dense_device=device,
        )
        train_model = DLRMTrain(dlrm_model)
        # This will apply the Adagrad optimizer in the backward pass for the embeddings (sparse_arch). This means that
        # the optimizer update will be applied in the backward pass, in this case through a fused op.
        # TorchRec will use the FBGEMM implementation of EXACT_ADAGRAD. For GPU devices, a fused CUDA kernel is invoked. For CPU, FBGEMM_GPU invokes CPU kernels
        # https://github.com/pytorch/FBGEMM/blob/2cb8b0dff3e67f9a009c4299defbd6b99cc12b8f/fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops.py#L676-L678
        apply_optimizer_in_backward(
            torch.optim.Adagrad,
            train_model.model.sparse_arch.parameters(),
            {"lr": args.learning_rate},
        )

        if args.shard_model:
            self.model = shard_modules(module=train_model, device=device).to(device)
        else:
            self.model = train_model.to(device)
        dense_optimizer = KeyedOptimizerWrapper(
            dict(in_backward_optimizer_filter(self.model.named_parameters())),
            lambda params: torch.optim.Adagrad(params, lr=args.learning_rate),
        )
        # fused optimizer will already be called
        opt = CombinedOptimizer([dense_optimizer])
        if args.multi_hot_sizes is not None:
            raise RuntimeError("Multi-hot is not supported in TorchBench.")

        if self.test == "train":
            self.opt = opt
            self.train_pipeline = TrainPipelineSparseDist(
                self.model,
                opt,
                device,
            )
            self.model.train()
        elif self.test == "eval":
            self.model.eval()

    def get_module(self):
        return self.model, (self.example_inputs,)

    def train(self):
        self.train_pipeline.progress(self.iterator)

    def eval(self):
        with torch.no_grad():
            _loss, logits = self.model(self.example_inputs)
        return logits
