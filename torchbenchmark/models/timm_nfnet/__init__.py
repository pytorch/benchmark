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

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

_logger = logging.getLogger('train')

class Model(BenchmarkModel):
    task = COMPUTER_VISION.CLASSIFICATION

    def __init__(self, device=None, jit=False, train_bs=128, eval_bs=256,
                 variant='dm_nfnet_f0'):
        super().__init__()
        self.device = device
        self.jit = jit

        # setup timm args
        args = get_args()
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
        self.args = args

        model, self.loader_train, self.loader_validate, self.optimizer, \
            self.train_loss_fn, self.lr_scheduler, self.amp_autocast, \
            self.loss_scaler, self.mixup_fn, self.validate_loss_fn = timm_instantiate_train(args)
        eval_model, self.loader_eval = timm_instantiate_eval(args)
        # jit the model if required
        self.model, self.eval_model = jit_if_needed(model, eval_model, jit=jit)
        
        # setup number of batches to run
        self.train_num_batch = 1
        self.eval_num_batch = 1

    def get_module(self):
        self.eval_model.eval()
        with torch.no_grad():
            for _, (input, _) in zip(range(self.eval_num_batch), self.loader_eval):
                return self.eval_model, (input, )

    def train(self, niter=1):
        self.model.train()
        eval_metric = self.args.eval_metric
        for epoch in range(niter):
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
                for _, (input, _) in zip(range(self.eval_num_batch), self.loader_eval):
                    if self.args.channels_last:
                        input = input.contiguous(memory_format=torch.channels_last)
                    with self.amp_autocast():
                        output = self.eval_model(input)