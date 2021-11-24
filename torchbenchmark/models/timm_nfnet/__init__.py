from timm.data.dataset_factory import create_dataset
import torch
from collections import OrderedDict
from contextlib import suppress

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION

# timm imports
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, convert_splitbn_model, safe_model_name, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler

from torchbenchmark.util.jit import jit_if_needed
from torchbenchmark.util.env_check import has_native_amp
from torchbenchmark.util.frameworks.timm.args import get_args

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

class Model(BenchmarkModel):
    task = COMPUTER_VISION.CLASSIFICATION

    def __init__(self, device=None, jit=False, variant='dm_nfnet_f0', precision='float32',
                 train_bs=128, eval_bs=256):
        super().__init__()
        self.device = device
        self.jit = jit

        # Setup args
        args = get_args()
        args.prefetcher = not args.no_prefetcher
        args.distributed = False
        args.world_size = 1
        args.rank = 0  # global rank
        # resolve AMP arguments based on PyTorch amp availability
        use_amp = None
        if args.amp and has_native_amp():
            use_amp = 'native'
        args.model_name = variant
        args.torchscript = jit

        # create train model
        model = create_model(
            args.model,
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
        # instantiate eval model
        eval_model = create_model(variant, pretrained=False, scriptable=True)
        
        # setup augmentation batch splits for contrastive loss or split bn
        num_aug_splits = 0
        if args.aug_splits > 0:
            assert args.aug_splits > 1, 'A split of 1 makes no sense'
            num_aug_splits = args.aug_splits

        # enable split bn (separate bn stats per batch-portion)
        if args.split_bn:
            assert num_aug_splits > 1 or args.resplit
            model = convert_splitbn_model(model, max(num_aug_splits, 2))
        model = model.to(device)
        eval_model = eval_model.to(device)
        # enable channels last layout if set
        if args.channels_last:
            model = model.to(memory_format=torch.channels_last)
            eval_model = eval_model.to(memory_format=torch.channels_last)

        self.model, self.eval_model = jit_if_needed(model, eval_model, jit=self.jit)

        # setup optimizer
        self.optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

        # setup automatic mixed-precision (AMP) loss scaling and op casting
        amp_autocast = suppress  # do nothing
        loss_scaler = None
        if use_amp == 'native':
            amp_autocast = torch.cuda.amp.autocast
            loss_scaler = NativeScaler()

        # setup scheduler
        lr_scheduler, num_epochs = create_scheduler(args, self.optimizer)

        # setup input
        data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)
        dataset_train = create_dataset()
        dataset_eval = create_dataset()

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

        # setup loader
        self.train_num_batch = 1
        self.eval_num_batch = 1

    def _step_eval(self):
        self.eval_model(self.cfg.infer_example_inputs)

    def get_module(self):
        return self.model, (self.cfg.example_inputs,)

    def train(self, niter=1):
        self.model.train()
        for _ in range(niter):
            train_one_epoch()

    # TODO: use pretrained model weights, assuming the pretrained model is in .data/ dir
    def eval(self, niter=1):
        self.eval_model.eval()
        with torch.no_grad():
            for _ in range(niter):
                self._step_eval()

if __name__ == "__main__":
    for device in ['cpu', 'cuda']:
        for jit in [False, True]:
            print("Test config: device %s, JIT %s" % (device, jit))
            m = Model(device=device, jit=jit)
            m, example_inputs = m.get_module()
            m(example_inputs)
            m.train()
            m.eval()
