"""
Reproduce Omniglot results of Snell et al Prototypical networks.
"""
from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse

from ..few_shot.datasets import OmniglotDataset
from ..few_shot.models import get_few_shot_encoder
from ..few_shot.core import NShotTaskSampler, EvaluateFewShot, prepare_nshot_task
from ..few_shot.proto import proto_net_episode
from ..few_shot.train import fit
from ..few_shot.callbacks import *
from ..few_shot.utils import setup_dirs

from ..config import PATH

class ProtoNets:
  def __init__(self, eval_bs = 1000, train_bs = 100, use_cuda = True, dataset='omniglot', distance='l2', n_train=1, n_test=1, k_train=60, k_test=5, q_train=5, q_test=1):
    torchbench = True

    setup_dirs()

    self.eval_bs = eval_bs

    if use_cuda:
      assert torch.cuda.is_available()
      self.device = torch.device('cuda')

    torch.backends.cudnn.benchmark = True

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

    self.evaluation_episodes = eval_bs #1000
    self.episodes_per_epoch = train_bs #100
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
      self.n_epochs = 3
      self.num_workers = 8
      self.persistent_workers = True

    self.param_str = f'{args.dataset}_nt={args.n_train}_kt={args.k_train}_qt={args.q_train}_' \
                     f'nv={args.n_test}_kv={args.k_test}_qv={args.q_test}'

    ###################
    # Create datasets #
    ###################
    background = self.dataset_class('background')
    self.background_taskloader = DataLoader(
      background,
      batch_sampler=NShotTaskSampler(background, self.episodes_per_epoch, args.n_train, args.k_train,   args.q_train),
      num_workers=self.num_workers,
      persistent_workers=self.persistent_workers
    )
    evaluation = self.dataset_class('evaluation')
    self.evaluation_taskloader = DataLoader(
      evaluation,
      batch_sampler=NShotTaskSampler(evaluation, self.episodes_per_epoch, self.args.n_test, self.args.k_test, self.args.q_test),
      num_workers=self.num_workers,
      persistent_workers=self.persistent_workers
    )

    #########
    # Model #
    #########
    self.model = get_few_shot_encoder(self.num_input_channels)
    self.model.to(self.device, dtype=torch.double)

    self.eval_fake = torch.randn(self.eval_bs, 1, 28, 28, device=self.device, dtype=torch.double)

  def get_module(self):
    return self.model, self.eval_fake

  def Eval(self, niter=200):
    self.model.eval()
    for _ in range(niter):
      x = self.model(self.eval_fake).view(-1)

    return
    

  def Train(self):
    ############
    # Training #
    ############
    self.model.train()
    #print(f'Training Prototypical network on {self.args.dataset}...')
    self.optimiser = Adam(self.model.parameters(), lr=1e-3)
    self.loss_fn = torch.nn.NLLLoss().cuda()

    def lr_schedule(epoch, lr):
      # Drop lr every 2000 episodes
      if epoch % self.drop_lr_every == 0:
          return lr / 2
      else:
          return lr

    self.callbacks = [
      EvaluateFewShot(
          eval_fn=proto_net_episode,
          num_tasks=self.evaluation_episodes,
          n_shot=self.args.n_test,
          k_way=self.args.k_test,
          q_queries=self.args.q_test,
          taskloader=self.evaluation_taskloader,
          prepare_batch=prepare_nshot_task(self.args.n_test, self.args.k_test, self.args.q_test),
          distance=self.args.distance
      ),
      ModelCheckpoint(
          filepath=PATH + f'/models/proto_nets/{self.param_str}.pth',
          monitor=f'val_{self.args.n_test}-shot_{self.args.k_test}-way_acc'
      ),
      LearningRateScheduler(schedule=lr_schedule),
      CSVLogger(PATH + f'/logs/proto_nets/{self.param_str}.csv'),
    ]

    fit(
      self.model,
      self.optimiser,
      self.loss_fn,
      epochs=self.n_epochs,
      dataloader=self.background_taskloader,
      prepare_batch=prepare_nshot_task(self.args.n_train, self.args.k_train, self.args.q_train),
      callbacks=self.callbacks,
      metrics=['categorical_accuracy'],
      fit_function=proto_net_episode,
      fit_function_kwargs={'n_shot': self.args.n_train, 'k_way': self.args.k_train, 'q_queries': self.args.q_train, 'train': True,
                           'distance': self.args.distance},
    )
