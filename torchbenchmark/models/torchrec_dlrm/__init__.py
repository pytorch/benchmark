import torch

# OSS import
try:
    # pyre-ignore[21]
    # @manual=//ai_codesign/benchmarks/dlrm/torchrec_dlrm/data:dlrm_dataloader
    from data.dlrm_dataloader import get_dataloader

    # pyre-ignore[21]
    # @manual=//ai_codesign/benchmarks/dlrm/torchrec_dlrm:lr_scheduler
    from lr_scheduler import LRPolicyScheduler

except ImportError:
    pass

from pyre_extensions import none_throws
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec import EmbeddingBagCollection
from torchrec.models.dlrm import DLRM, DLRM_DCN, DLRM_Projection, DLRMTrain
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import RECOMMENDATION

from .args import parse_args, InteractionType


class Model(BenchmarkModel):
    task = RECOMMENDATION.RECOMMENDATION
    # upstream default batch size: 32
    # https://github.com/facebookresearch/dlrm/blob/main/torchrec_dlrm/dlrm_main.py#L86
    DEFAULT_TRAIN_BSIZE = 32
    DEFAULT_EVAL_BSIZE = 32

    def __init__(self, test, device, jit, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        args = parse_args(self.extra_args)

        train_dataloader = None
        eb_configs = [
            EmbeddingBagConfig(
                name=f"t_{feature_name}",
                embedding_dim=args.embedding_dim,
                num_embeddings=none_throws(args.num_embeddings_per_feature)[feature_idx]
                if args.num_embeddings is None
                else args.num_embeddings,
                feature_names=[feature_name],
            )
            for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
        ]
        if args.interaction_type == InteractionType.ORIGINAL:
            dlrm_model = DLRM(
                embedding_bag_collection=EmbeddingBagCollection(
                    tables=eb_configs, device=torch.device("meta")
                ),
                dense_in_features=len(DEFAULT_INT_NAMES),
                dense_arch_layer_sizes=args.dense_arch_layer_sizes,
                over_arch_layer_sizes=args.over_arch_layer_sizes,
                dense_device=device,
            )
        elif args.interaction_type == InteractionType.DCN:
            dlrm_model = DLRM_DCN(
                embedding_bag_collection=EmbeddingBagCollection(
                    tables=eb_configs, device=torch.device("meta")
                ),
                dense_in_features=len(DEFAULT_INT_NAMES),
                dense_arch_layer_sizes=args.dense_arch_layer_sizes,
                over_arch_layer_sizes=args.over_arch_layer_sizes,
                dcn_num_layers=args.dcn_num_layers,
                dcn_low_rank_dim=args.dcn_low_rank_dim,
                dense_device=device,
            )
        elif args.interaction_type == InteractionType.PROJECTION:
            dlrm_model = DLRM_Projection(
                embedding_bag_collection=EmbeddingBagCollection(
                    tables=eb_configs, device=torch.device("meta")
                ),
                dense_in_features=len(DEFAULT_INT_NAMES),
                dense_arch_layer_sizes=args.dense_arch_layer_sizes,
                over_arch_layer_sizes=args.over_arch_layer_sizes,
                interaction_branch1_layer_sizes=args.interaction_branch1_layer_sizes,
                interaction_branch2_layer_sizes=args.interaction_branch2_layer_sizes,
                dense_device=device,
            )
        else:
            raise ValueError(
                "Unknown interaction option set. Should be original, dcn, or projection."
            )
        train_model = DLRMTrain(dlrm_model)
        embedding_optimizer = torch.optim.Adagrad if args.adagrad else torch.optim.SGD
        # This will apply the Adagrad optimizer in the backward pass for the embeddings (sparse_arch). This means that
        # the optimizer update will be applied in the backward pass, in this case through a fused op.
        # TorchRec will use the FBGEMM implementation of EXACT_ADAGRAD. For GPU devices, a fused CUDA kernel is invoked. For CPU, FBGEMM_GPU invokes CPU kernels
        # https://github.com/pytorch/FBGEMM/blob/2cb8b0dff3e67f9a009c4299defbd6b99cc12b8f/fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops.py#L676-L678
        apply_optimizer_in_backward(
            embedding_optimizer,
            train_model.model.sparse_arch.parameters(),
            {"lr": args.learning_rate},
        )
        planner = EmbeddingShardingPlanner(
            topology=Topology(
                local_world_size=get_local_size(),
                world_size=dist.get_world_size(),
                compute_device=device.type,
            ),
            batch_size=args.batch_size,
            # If experience OOM, increase the percentage. see
            # https://pytorch.org/torchrec/torchrec.distributed.planner.html#torchrec.distributed.planner.storage_reservations.HeuristicalStorageReservation
            storage_reservation=HeuristicalStorageReservation(percentage=0.05),
        )
        plan = planner.collective_plan(
            train_model, get_default_sharders(), dist.GroupMember.WORLD
        )
        def optimizer_with_params():
            if args.adagrad:
                return lambda params: torch.optim.Adagrad(params, lr=args.learning_rate)
            else:
                return lambda params: torch.optim.SGD(params, lr=args.learning_rate)

        dense_optimizer = KeyedOptimizerWrapper(
            dict(in_backward_optimizer_filter(model.named_parameters())),
            optimizer_with_params(),
        )
        optimizer = CombinedOptimizer([model.fused_optimizer, dense_optimizer])
        self.lr_scheduler = LRPolicyScheduler(
            optimizer, args.lr_warmup_steps, args.lr_decay_start, args.lr_decay_steps
        )
        if args.multi_hot_sizes is not None:
            raise RuntimeError("Multi-hot is not supported in TorchBench.")


    def train(self):
        pass

    def eval(self):
        pass
