import os
import logging
import torch
from pathlib import Path
from contextlib import suppress

# TorchBench imports
from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION

# effdet imports
from effdet import create_model, create_loader
from effdet.data import resolve_input_config

# timm imports
from timm.models.layers import set_layer_config
from timm.optim import create_optimizer
from timm.utils import ModelEmaV2, NativeScaler
from timm.scheduler import create_scheduler

# local imports
from .args import get_args
from .train import train_epoch, validate
from .loader import create_datasets_and_loaders

from typing import Tuple

# setup coco2017 input path
CURRENT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(CURRENT_DIR.parent.parent, "data", ".data", "coco2017-minimal", "coco")

class Model(BenchmarkModel):
    task = COMPUTER_VISION.DETECTION
    # Original Train batch size 32 on 2x RTX 3090 (24 GB cards)
    # Downscale to batch size 16 on single GPU
    DEFAULT_TRAIN_BSIZE = 16
    DEFAULT_EVAL_BSIZE = 128

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        if not device == "cuda":
            # Only implemented on CUDA because the original model code explicitly calls the `Tensor.cuda()` API
            # https://github.com/rwightman/efficientdet-pytorch/blob/9cb43186711d28bd41f82f132818c65663b33c1f/effdet/data/loader.py#L114
            raise NotImplementedError("The original model code forces the use of CUDA.")
        # generate arguments
        args = get_args()
        # setup train and eval batch size
        args.batch_size = self.batch_size
        # Disable distributed
        args.distributed = False
        args.device = self.device
        args.torchscript = self.jit
        args.world_size = 1
        args.rank = 0
        args.pretrained_backbone = not args.no_pretrained_backbone
        args.prefetcher = not args.no_prefetcher
        args.root = DATA_DIR

        with set_layer_config(scriptable=args.torchscript):
            timm_extra_args = {}
            if args.img_size is not None:
                timm_extra_args = dict(image_size=(args.img_size, args.img_size))
            if test == "train":
                model = create_model(
                    model_name=args.model,
                    bench_task='train',
                    num_classes=args.num_classes,
                    pretrained=args.pretrained,
                    pretrained_backbone=args.pretrained_backbone,
                    redundant_bias=args.redundant_bias,
                    label_smoothing=args.smoothing,
                    legacy_focal=args.legacy_focal,
                    jit_loss=args.jit_loss,
                    soft_nms=args.soft_nms,
                    bench_labeler=args.bench_labeler,
                    checkpoint_path=args.initial_checkpoint,
                )
            elif test == "eval":
                model = create_model(
                    model_name=args.model,
                    bench_task='predict',
                    num_classes=args.num_classes,
                    pretrained=args.pretrained,
                    redundant_bias=args.redundant_bias,
                    soft_nms=args.soft_nms,
                    checkpoint_path=args.checkpoint,
                    checkpoint_ema=args.use_ema,
                    **timm_extra_args,
                )
        model_config = model.config  # grab before we obscure with DP/DDP wrappers
        self.model = model.to(device)
        if args.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.loader_train, self.loader_eval, self.evaluator, _, dataset_eval = create_datasets_and_loaders(args, model_config)
        self.amp_autocast = suppress

        if test == "train":
            self.optimizer = create_optimizer(args, model)
            self.loss_scaler = None
            self.model_ema = None
            if args.model_ema:
                # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
                self.model_ema = ModelEmaV2(model, decay=args.model_ema_decay)
            self.lr_scheduler, self.num_epochs = create_scheduler(args, self.optimizer)
            if model_config.num_classes < self.loader_train.dataset.parser.max_label:
                logging.error(
                    f'Model {model_config.num_classes} has fewer classes than dataset {self.loader_train.dataset.parser.max_label}.')
                exit(1)
            if model_config.num_classes > self.loader_train.dataset.parser.max_label:
                logging.warning(
                    f'Model {model_config.num_classes} has more classes than dataset {self.loader_train.dataset.parser.max_label}.')
        elif test == "eval":
            # Create eval loader
            input_config = resolve_input_config(args, model_config)
            self.loader = create_loader(
                    dataset_eval,
                    input_size=input_config['input_size'],
                    batch_size=args.batch_size,
                    use_prefetcher=args.prefetcher,
                    interpolation=args.eval_interpolation,
                    fill_color=input_config['fill_color'],
                    mean=input_config['mean'],
                    std=input_config['std'],
                    num_workers=args.workers,
                    pin_mem=args.pin_mem)
        self.args = args
        # Only run 1 batch in 1 epoch
        self.num_batches = 1
        self.num_epochs = 1

    def get_module(self):
        for _, (input, target) in zip(range(self.num_batches), self.loader):
            return self.model, (input, target)

    def enable_amp(self):
        self.amp_autocast = torch.cuda.amp.autocast
        self.loss_scaler = NativeScaler()

    def train(self):
        eval_metric = self.args.eval_metric
        for epoch in range(self.num_epochs):
            train_metrics = train_epoch(
                epoch, self.model, self.loader_train,
                self.optimizer, self.args,
                lr_scheduler=self.lr_scheduler, amp_autocast = self.amp_autocast,
                loss_scaler=self.loss_scaler, model_ema=self.model_ema,
                num_batch=self.num_batches,
            )
            # the overhead of evaluating with coco style datasets is fairly high, so just ema or non, not both
            if self.model_ema is not None:
                eval_metrics = validate(self.model_ema.module, self.loader_eval, self.args, self.evaluator, log_suffix=' (EMA)', num_batch=self.num_batches)
            else:
                eval_metrics = validate(self.model, self.loader_eval, self.args, self.evaluator, num_batch=self.num_batches)
            if self.lr_scheduler is not None:
                # step LR for next epoch
                self.lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

    def eval(self) -> Tuple[torch.Tensor]:
        with torch.no_grad():
            for _, (input, target) in zip(range(self.num_batches), self.loader):
                with self.amp_autocast():
                    output = self.model(input, img_info=target)
                self.evaluator.add_predictions(output, target)
        return (output, )
