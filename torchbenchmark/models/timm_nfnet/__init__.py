import torch
import os

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION
import logging

from torchbenchmark.util.env_check import has_native_amp
from torchbenchmark.util.framework.timm.args import get_args, setup_args_distributed
from torchbenchmark.util.framework.timm.train import train_one_epoch, validate
from torchbenchmark.util.framework.timm.instantiate import timm_instantiate_train, timm_instantiate_eval

from torchbenchmark.util.jit import jit_if_needed
from torchbenchmark.util.prefetch import prefetch_loader

from torchbenchmark.util.framework.timm.extra_args import parse_args_nfnet, apply_args_nfnet

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

_logger = logging.getLogger('train')

TRAIN_NUM_BATCH = 1
EVAL_NUM_BATCH = 1

class Model(BenchmarkModel):
    task = COMPUTER_VISION.CLASSIFICATION

    # Original train batch size 128, hardware Nvidia rtx 3090
    # Source: https://gist.github.com/rwightman/bb59f9e245162cee0e38bd66bd8cd77f#file-bench_by_train-csv-L147
    # Eval batch size 256, hardware Nvidia rtx 3090
    # Source: https://github.com/rwightman/pytorch-image-models/blob/f7d210d759beb00a3d0834a3ce2d93f6e17f3d38/results/model_benchmark_amp_nchw_rtx3090.csv
    # Downscale to 128 to fit T4
    def __init__(self, device=None, jit=False, train_bs=128, eval_bs=128,
                 variant='dm_nfnet_f0', extra_args=[]):
        super().__init__()
        self.device = device
        self.jit = jit
        self.train_bs = train_bs
        self.eval_bs = eval_bs

        # setup timm args
        args = get_args()
        # use fp16 by default for both train and eval
        args.amp = True
        args.torchscript = jit
        args.device = device
        # setup distributed
        args = setup_args_distributed(args)
        if args.distributed:
            _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                          % (args.rank, args.world_size))
        else:
            _logger.info('Testing with a single process on 1 GPU.')
        if not device or device == "cpu":
            args.prefetcher = False
        else:
            args.prefetcher = not args.no_prefetcher
            assert args.prefetcher, "Test requires the data to be prefetched during execution"
        # resolve AMP arguments based on PyTorch amp availability
        args.use_amp = None
        if args.amp and has_native_amp():
            args.use_amp = 'native'
        args.model_name = variant
        args.batch_size = train_bs
        args.eval_batch_size = eval_bs
        args.train_num_batch = TRAIN_NUM_BATCH
        args.eval_num_batch = EVAL_NUM_BATCH
        self.args = args

        model, self.loader_train, self.loader_validate, self.optimizer, \
            self.train_loss_fn, self.lr_scheduler, self.amp_autocast, \
            self.loss_scaler, self.mixup_fn, self.validate_loss_fn = timm_instantiate_train(args)
        eval_model, self.eval_example_inputs = timm_instantiate_eval(args)
        # jit the model if required
        self.model, self.eval_model = jit_if_needed(model, eval_model, jit=jit)
        
        # Disable train data load as there is not enough GPU memory
        # TODO: enable with larger GPU
        # self.loader_train = prefetch_loader(self.loader_train, device)
        # self.loader_validate = prefetch_loader(self.loader_validate, device)
        self.eval_example_inputs = prefetch_loader(self.eval_example_inputs, device)

        self.extra_args = parse_args_nfnet(self, extra_args)
        apply_args_nfnet(self, self.extra_args)

    def get_module(self):
        if self.device == "cuda":
            raise NotImplementedError("Disable get_module() because it causes CUDA OOM on Nvidia T4")
        self.eval_model.eval()
        with torch.no_grad():
            for _, (input, _) in zip(range(self.args.eval_num_batch), self.loader_eval):
                return self.eval_model, (input, )

    # Temporarily disable training because this will cause CUDA OOM in CI
    # TODO: re-enable this test when better hardware is available
    def train(self, niter=1):
        if self.device == "cuda":
            raise NotImplementedError("Disable the train test because it causes CUDA OOM on Nvidia T4")
        self.model.train()
        eval_metric = self.args.eval_metric
        for epoch in range(niter):
            # run "train_num_batch" batches per epoch
            train_metrics = train_one_epoch(epoch, self.model, self.loader_train,
                                            self.optimizer, self.train_loss_fn, self.args,
                                            lr_scheduler=self.lr_scheduler, saver=None,
                                            output_dir=None,
                                            amp_autocast=self.amp_autocast,
                                            loss_scaler=self.loss_scaler,
                                            model_ema=None,
                                            mixup_fn=self.mixup_fn)
            eval_metrics = validate(self.model, self.loader_validate, self.validate_loss_fn,
                                    self.args, amp_autocast=self.amp_autocast)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch+1, eval_metrics[eval_metric])

    # We skipped computing loss and accuracy in eval
    def eval(self, niter=1):
        self.eval_model.eval()
        for epoch in range(niter):
            with torch.no_grad():
                for _, (input, _) in zip(range(self.args.eval_num_batch), self.loader_eval):
                    if self.args.channels_last:
                        input = input.contiguous(memory_format=torch.channels_last)
                    with self.amp_autocast():
                        output = self.eval_model(input)
