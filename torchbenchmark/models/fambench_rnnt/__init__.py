"""
TorchBench version of FAMBench RNNT model
For train test, optimizer is only enabled when the amp is enabled and apex module is available.
"""
import torch
import sys
import os
import copy
import random
import toml
from torchbenchmark import REPO_PATH
import numpy as np
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
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
RNNT_TRAIN_PATH = os.path.join(CURRENT_DIR)
RNNT_EVAL_PATH = os.path.join(REPO_PATH, "submodules", "FAMBench", "benchmarks", "rnnt", "ootb", "inference", "pytorch")

with add_path(RNNT_TRAIN_PATH):
    from rnnt import config
    from rnnt.loss import RNNTLoss
    from rnnt.model import RNNT as RNNTTrain
    from common.data.text import Tokenizer
    from common.data import features
    from common.data.dali import sampler as dali_sampler
    from common.data.dali.data_loader import DaliDataLoader
    # lr policy is not supported
    # from common.optimizers import lr_policy

with add_path(RNNT_EVAL_PATH):
    from helpers import add_blank_label
    from preprocessing import AudioPreprocessing
    from model_separable_rnnt import RNNT as RNNTEval

from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.tasks import SPEECH
from .qsl import AudioQSLInMemory
from .args import get_eval_args, get_train_args
from .config import FambenchRNNTTrainConfig, FambenchRNNTEvalConfig, cfg_to_str
from .decoders import ScriptGreedyDecoder
from .utils import apply_ema

class Model(BenchmarkModel):
    task = SPEECH.RECOGNITION
    RNNT_TRAIN_CONFIG = FambenchRNNTTrainConfig()
    RNNT_EVAL_CONFIG = FambenchRNNTEvalConfig()

    # This model doesn't allow customize batch size
    DEFAULT_TRAIN_BATCH_SIZE = RNNT_TRAIN_CONFIG.batch_size
    DEFAULT_EVAL_BATCH_SIZE = RNNT_EVAL_CONFIG.batch_size
    ALLOW_CUSTOMIZE_BSIZE = False
    # run only 1 batch
    DEFAULT_NUM_BATCHES = 1

    def __init__(self):
        if self.test == "train":
            self._init_train()
        elif self.test == "eval":
            self._init_eval()


    def train(self):
        # torchbench: distributed training is not supported in core models
        world_size = 1
        for batch in self.dl:
            audio, audio_lens, txt, txt_lens = batch

            feats, feat_lens = self.train_feat_proc([audio, audio_lens])
            all_feat_lens += feat_lens
            log_probs, log_prob_lens = self.model(feats, feat_lens, txt, txt_lens)
            loss = self.loss_fn(log_probs[:, :log_prob_lens.max().item()],
                           log_prob_lens, txt, txt_lens)
            loss /= self.fambench_args.grad_accumulation_steps

            del log_probs, log_prob_lens
            if torch.isnan(loss).any():
                print('WARNING: loss is NaN; skipping update')
            else:
                if self.fambench_args.amp:
                    from apex import amp
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                loss_item = loss.item()
                del loss
                step_utts += batch[0].size(0) * world_size
                epoch_utts += batch[0].size(0) * world_size
                accumulated_batches += 1
                total_batches += 1
            if accumulated_batches % self.fambench_args.grad_accumulation_steps == 0:
                total_norm = 0.0
                try:
                    if self.fambench_args.log_norm:
                        for p in getattr(self.model, 'module', self.model).parameters():
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** (1. / 2)
                except AttributeError as e:
                    print(f'Exception happened: {e}')
                    total_norm = 0.0
                if self.fambench_args.amp:
                    self.optimizer.step()
                apply_ema(self.model, self.ema_model, self.fambench_args.ema)

    # reference: FAMBench/benchmarks/rnnt/ootb/inference/pytorch_SUT.py
    def eval(self) -> Tuple[torch.Tensor]:
        for query_sample in self.qsl.sample_id_to_sample:
            waveform = query_sample
            assert waveform.ndim == 1
            waveform_length = np.array(waveform.shape[0], dtype=np.int64)
            waveform = np.expand_dims(waveform, 0)
            waveform_length = np.expand_dims(waveform_length, 0)
            with torch.no_grad():
                waveform = torch.from_numpy(waveform)
                waveform_length = torch.from_numpy(waveform_length)
                feature, feature_length = self.audio_preprocessor.forward((waveform, waveform_length))
                assert feature.ndim == 3
                assert feature_length.ndim == 1
                feature = feature.permute(2, 0, 1)

                _, _, transcript = self.greedy_decoder.forward(feature, feature_length)

    def enable_jit(self):
        if self.test == "eval":
            self.audio_preprocessor = torch.jit.script(self.audio_preprocessor)
            self.audio_preprocessor = torch.jit._recursive.wrap_cpp_module(
                torch._C._freeze_module(self.audio_preprocessor._c))
            self.innner_model.encoder = torch.jit.script(self.inner_model.encoder)
            self.inner_model.encoder = torch.jit._recursive.wrap_cpp_module(
                torch._C._freeze_module(self.inner_model.encoder._c))
            self.inner_model.prediction = torch.jit.script(self.inner_model.prediction)
            self.inner_model.prediction = torch.jit._recursive.wrap_cpp_module(
                torch._C._freeze_module(self.inner_model.prediction._c))
            self.inner_model.joint = torch.jit.script(self.inner_model.joint)
            self.inner_model.joint = torch.jit._recursive.wrap_cpp_module(
                torch._C._freeze_module(self.inner_model.joint._c))
            self.inner_model = torch.jit.script(self.inner_model)
            self.greedy_decoder.set_model(self.inner_model)
        elif self.test == "train":
            self.model = torch.jit.script(self.model)

    def _init_train(self):
        args = cfg_to_str(self.RNNT_TRAIN_CONFIG)
        args = get_train_args(args)
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        world_size = 1
        if args.seed is not None:
            torch.manual_seed(args.seed + args.local_rank)
            np.random.seed(args.seed + args.local_rank)
            random.seed(args.seed + args.local_rank)
            # np_rng is used for buckets generation, and needs the same seed on every worker
            np_rng = np.random.default_rng(seed=args.seed)
        cfg = config.load(args.model_config)
        config.apply_duration_flags(cfg, args.max_duration)
        assert args.grad_accumulation_steps >= 1
        assert args.batch_size % args.grad_accumulation_steps == 0, \
            f'{args.batch_size} % {args.grad_accumulation_steps} != 0'
        batch_size = args.batch_size // args.grad_accumulation_steps
        (
            train_dataset_kw,
            train_features_kw,
            train_splicing_kw,
            train_specaugm_kw,
        ) = config.input(cfg, 'train')
        tokenizer_kw = config.tokenizer(cfg)
        tokenizer = Tokenizer(**tokenizer_kw)

        class PermuteAudio(torch.nn.Module):
            def forward(self, x):
                return (x[0].permute(2, 0, 1), *x[1:])

        train_augmentations = torch.nn.Sequential(
            train_specaugm_kw and features.SpecAugment(optim_level=args.amp, **train_specaugm_kw) or torch.nn.Identity(),
            features.FrameSplicing(optim_level=args.amp, **train_splicing_kw),
            PermuteAudio(),
        )
        if args.num_buckets is not None:
            sampler = dali_sampler.BucketingSampler(
                args.num_buckets,
                batch_size,
                world_size,
                args.epochs,
                np_rng
            )
        else:
            sampler = dali_sampler.SimpleSampler()
        train_loader = DaliDataLoader(gpu_id=args.local_rank,
                                dataset_path=args.dataset_dir,
                                config_data=train_dataset_kw,
                                config_features=train_features_kw,
                                json_names=args.train_manifests,
                                batch_size=batch_size,
                                sampler=sampler,
                                grad_accumulation_steps=args.grad_accumulation_steps,
                                pipeline_type="train",
                                device_type=args.dali_device,
                                tokenizer=tokenizer)
        train_feat_proc = train_augmentations
        train_feat_proc.to(self.device)
        # steps_per_epoch = len(train_loader) // args.grad_accumulation_steps

        # setup model
        rnnt_config = config.rnnt(cfg)
        if args.weights_init_scale is not None:
            rnnt_config['weights_init_scale'] = args.weights_init_scale
        if args.hidden_hidden_bias_scale is not None:
            rnnt_config['hidden_hidden_bias_scale'] = args.hidden_hidden_bias_scale
        model = RNNTTrain(n_classes=tokenizer.num_labels + 1, **rnnt_config)
        model = model.to(self.device)
        blank_idx = tokenizer.num_labels
        loss_fn = RNNTLoss(blank_idx=blank_idx, device=self.device)
        opt_eps = 1e-9
        # optimization
        kw = {'params': model.param_groups(args.lr), 'lr': args.lr,
            'weight_decay': args.weight_decay}
        # initial_lrs = [group['lr'] for group in kw['params']]
        if args.amp:
            from apex.optimizers import FusedLAMB
            optimizer = FusedLAMB(betas=(args.beta1, args.beta2), eps=opt_eps, max_grad_norm=args.clip_norm, **kw)

        # adjust_lr = lambda step, epoch: lr_policy(
        #     step, epoch, initial_lrs, optimizer, steps_per_epoch=steps_per_epoch,
        #     warmup_epochs=args.warmup_epochs, hold_epochs=args.hold_epochs,
        #     min_lr=args.min_lr, exp_gamma=args.lr_exp_gamma)
        if args.amp:
            from apex import amp
            model, optimizer = amp.initialize(models=self.model,
                                                optimizers=self.optimizer,
                                                opt_level='01',
                                                max_loss_scale=512.0)
        if args.ema > 0:
            ema_model = copy.deepcopy(model).cuda()
        else:
            ema_model = None
        # checkpoint is not supported
        assert args.ckpt == None, "Checkpointing is not supported in TorchBench"
        model.train()
        # members used in train loop
        self.fambench_args = args
        self.model = model
        self.optimizer = optimizer
        self.ema_model = ema_model
        self.loss_fn = loss_fn
        self.train_feat_proc = train_feat_proc

    def enable_amp(self):
        if self.test == "train":
            self.RNNT_TRAIN_CONFIG.amp = "true"
            self._init_train()
        elif self.test == "eval":
            raise NotImplementedError("FAMBench rnnt doesn't support AMP inference.")

    def _init_eval(self):
        args = cfg_to_str(self.RNNT_EVAL_CONFIG)
        args = get_eval_args(args) 
        assert args.backend == "pytorch", f"Unknown backend: {args.backend}"
        config = toml.load(args.config_toml)

        dataset_vocab = config['labels']['labels']
        rnnt_vocab = add_blank_label(dataset_vocab)
        featurizer_config = config['input_eval']

        self.qsl = AudioQSLInMemory(args.dataset_dir,
                                    args.manifest_filepath,
                                    dataset_vocab,
                                    featurizer_config["sample_rate"],
                                    args.perf_count)
        self.audio_preprocessor = AudioPreprocessing(**featurizer_config)
        self.audio_preprocessor.eval()

        model = RNNTEval(
            feature_config=featurizer_config,
            rnnt=config['rnnt'],
            num_classes=len(rnnt_vocab)
        )
        # torchbench: we don't support load checkpoint state dict
        # model.load_state_dict(load_and_migrate_checkpoint(checkpoint_path),
        #                       strict=True)
        model.eval()

        self.inner_model = model
        self.greedy_decoder = ScriptGreedyDecoder(len(rnnt_vocab) - 1, self.inner_model)