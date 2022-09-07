"""
Reproduce Omniglot results of Snell et al Prototypical networks.
"""
from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse

from ..few_shot.datasets import OmniglotDataset, MiniImageNet
from ..few_shot.models import get_few_shot_encoder
from ..few_shot.core import NShotTaskSampler, EvaluateFewShot, prepare_nshot_task
from ..few_shot.proto import proto_net_episode
from ..few_shot.train import fit
from ..few_shot.callbacks import *
from ..few_shot.utils import setup_dirs
from torch.utils._pytree import tree_map

def prefetch(dataloader, device, precision="fp32"):
    r = []
    dtype = torch.float16 if precision == "fp16" else torch.float32
    for batch in dataloader:
        r.append(tree_map(lambda x: x.to(device, dtype=dtype) if isinstance(x, torch.Tensor) else x, batch))
    return r

from ..config import PATH

class ProtoNets:
  def __init__(self, test, bs, num_of_batches, device, dataset='omniglot', distance='l2', n_train=1, n_test=1, k_train=60, k_test=5, q_train=5, q_test=1):
    torchbench = True

    # setup model runtime directories
    setup_dirs()

    self.test = test
    self.bs = bs
    self.device = device

    ##############
    # Parameters #
    ##############

    class Args:
      pass

    args = Args()

    args.torchbench = torchbench

    if torchbench:
      args.dataset=dataset
      args.distance=distance
      args.n_train=n_train
      args.n_test=n_test
      args.k_train=k_train
      args.k_test=k_test
      args.q_train=q_train
      args.q_test=q_test
    else:
      parser = argparse.ArgumentParser()
      parser.add_argument('--dataset')
      parser.add_argument('--distance', default='l2')
      parser.add_argument('--n-train', default=1, type=int)
      parser.add_argument('--n-test', default=1, type=int)
      parser.add_argument('--k-train', default=60, type=int)
      parser.add_argument('--k-test', default=5, type=int)
      parser.add_argument('--q-train', default=5, type=int)
      parser.add_argument('--q-test', default=1, type=int)
      parser.add_argument('--torchbench', dest='torchbench', action='store_true')
      parser.set_defaults(torchbench=False)
      args = parser.parse_args()

    self.args = args
    self.evaluation_episodes = num_of_batches
    self.episodes_per_epoch = num_of_batches
    self.persistent_workers = True
    self.num_workers = 4

    if args.dataset == 'omniglot':
      self.n_epochs = 40
      self.dataset_class = OmniglotDataset
      self.num_input_channels = 1
      self.drop_lr_every = 20
    elif args.dataset == 'miniImageNet':
      self.n_epochs = 80
      self.dataset_class = MiniImageNet
      self.num_input_channels = 3
      self.drop_lr_every = 40
    else:
      raise(ValueError, 'Unsupported dataset')

    if args.torchbench:
      # torchbench: benchmark only 1 epoch
      self.n_epochs = 1
      self.num_workers = 8
      self.persistent_workers = True

    self.param_str = f'{args.dataset}_nt={args.n_train}_kt={args.k_train}_qt={args.q_train}_' \
                     f'nv={args.n_test}_kv={args.k_test}_qv={args.q_test}'

    #########
    # Model #
    #########
    self.model = get_few_shot_encoder(self.num_input_channels)
    self.model.to(self.device, dtype=torch.double)

    ###################
    # Create datasets #
    ###################
    if self.test == "train":
      background = self.dataset_class('evaluation')
      self.dl = DataLoader(
        background,
        batch_sampler=NShotTaskSampler(background, self.episodes_per_epoch, args.n_train, args.k_train, args.q_train, num_tasks=bs),
        num_workers=self.num_workers,
        persistent_workers=self.persistent_workers
      )
      self.prepare_batch = prepare_nshot_task(args.n_train, args.k_train, args.q_train)
      self.fit_function = proto_net_episode
      self.fit_function_kwargs = {'n_shot': args.n_train, 'k_way': args.k_train, 'q_queries': args.q_train,
                                  'train': True, 'distance': args.distance}
      self.model.train()
      self.optimiser = Adam(self.model.parameters(), lr=1e-3)
      self.loss_fn = torch.nn.NLLLoss().to(self.device)
    elif self.test == "eval":
      evaluation = self.dataset_class('evaluation')
      self.dl = DataLoader(
        evaluation,
        batch_sampler=NShotTaskSampler(evaluation, self.episodes_per_epoch, self.args.n_test, self.args.k_test, self.args.q_test, num_tasks=bs),
        num_workers=self.num_workers,
        persistent_workers=self.persistent_workers
      )
      self.loss_fn = torch.nn.NLLLoss().to(self.device)
      self.prepare_batch = prepare_nshot_task(self.args.n_test, self.args.k_test, self.args.q_test)
      self.model.eval()
    self.dl = prefetch(self.dl, device)

  def get_module(self):
    return self.model, (self.dl[0], )

  def Eval(self):
    with torch.no_grad():
      for batch in self.dl:
        x, y = self.prepare_batch(batch)
        y_pred = self.model(x)
        if self.loss_fn is not None:
           self.loss_fn(y_pred, y).item() * x.shape[0]

  def Train(self):
    ############
    # Training #
    ############
    for batch_index, batch in enumerate(self.dl):
        batch_logs = dict(batch=batch_index, size=(self.bs or 1))
        x, y = self.prepare_batch(batch)

        loss, y_pred = self.fit_function(self.model, self.optimiser, self.loss_fn, x, y, **self.fit_function_kwargs)
        batch_logs['loss'] = loss.item()

