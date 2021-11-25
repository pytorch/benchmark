""" Hacked from https://github.com/rwightman/pytorch-image-models/blob/f7d210d759beb00a3d0834a3ce2d93f6e17f3d38/train.py

ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""

from contextlib import suppress
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.models import create_model, convert_splitbn_model
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.data import create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset

from .loader import create_fake_imagenet_dataset

def timm_instantiate_eval(args):
    # create eval model
    eval_model = create_model(
        args.model_name,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=3,
        global_pool=args.gp,
        scriptable=args.torchscript)
    data_config = resolve_data_config(vars(args), model=eval_model, use_test_size=True, verbose=True)
    eval_model = eval_model.to(args.device)
    # enable channels last layout if set
    if args.channels_last:
        eval_model = eval_model.to(memory_format=torch.channels_last)
    if args.num_gpu > 1:
        eval_model = torch.nn.DataParallel(eval_model, device_ids=list(range(args.num_gpu)))
    crop_pct = data_config['crop_pct']
    # create dataset
    dataset_eval = create_fake_imagenet_dataset(size=args.eval_num_batch*args.eval_batch_size)
    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.eval_batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=crop_pct,
        pin_memory=args.pin_mem,
        tf_preprocessing=args.tf_preprocessing,
        persistent_workers=False,
    )
    return eval_model, loader_eval

def timm_instantiate_train(args):
    # create train model
    model = create_model(
        args.model_name,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint)
    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)
    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits
    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))
    model = model.to(args.device)
    # enable channels last layout if set
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            print(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    # setup optimizer
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if args.use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    # setup distributed training
    if args.distributed:
        model = NativeDDP(model, device_ids=[args.local_rank], broadcast_buffers=not args.no_ddp_bb)
        # NOTE: EMA model does not need to be wrapped by DDP

    # setup learning rate schedule and starting epoch
    lr_scheduler, _ = create_scheduler(args, optimizer)

    # create fake imagenet dataset
    fake_dataset = create_fake_imagenet_dataset(size=args.batch_size * args.train_num_batch)
    dataset_train = fake_dataset
    dataset_eval = fake_dataset

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeline
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        # Not supported by timm 0.4.12
        # num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        # Not supported by timm 0.4.12
        # worker_seeding=args.worker_seeding,
        persistent_workers=False,
    )

    loader_validate = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
        persistent_workers=False,
    )

    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # NOTE: the latest timm package (0.4.12) doesn't support BinaryCrossEntropy
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        # if args.bce_loss:
        #     train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        # else:
        train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        # if args.bce_loss:
        #     train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        # else:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.to(args.device)
    validate_loss_fn = nn.CrossEntropyLoss().to(args.device)

    # return all the inputs needed by train and eval loop
    return model, loader_train, loader_validate, optimizer, \
        train_loss_fn, lr_scheduler, amp_autocast, \
        loss_scaler, mixup_fn, validate_loss_fn
