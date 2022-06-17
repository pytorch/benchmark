"""
Reproduce Matching Network results of Vinyals et al
"""
import argparse
from torch.utils.data import DataLoader
from torch.optim import Adam

from ..few_shot.datasets import OmniglotDataset
from ..few_shot.core import NShotTaskSampler, prepare_nshot_task, EvaluateFewShot
from ..few_shot.matching import matching_net_episode
from ..few_shot.train import fit
from ..few_shot.callbacks import *
from ..few_shot.utils import setup_dirs
from ..config import PATH

class MatchingNets:
  def __init__(self, eval_bs = 1000, train_bs = 100, use_cuda = True, dataset='omniglot', distance='l2', n_train=1, n_test=1, k_train=60, k_test=5, q_train=5, q_test=1, lstm_layers=1, unrolling_steps=2, fce=False):
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
      args.dataset  = dataset
      args.distance = distance
      args.n_train  = n_train
      args.n_test   = n_test
      args.k_train  = k_train
      args.k_test   = k_test
      args.q_train  = q_train
      args.q_test   = q_test
      args.lstm_layers = lstm_layers
      args.fce      = fce
      args.unrolling_steps = unrolling_steps

    else:
      parser = argparse.ArgumentParser()
      parser.add_argument('--dataset')
      parser.add_argument('--fce', type=lambda x: x.lower()[0] == 't')  # Quick hack to extract boolean
      parser.add_argument('--distance', default='cosine')
      parser.add_argument('--n-train', default=1, type=int)
      parser.add_argument('--n-test', default=1, type=int)
      parser.add_argument('--k-train', default=5, type=int)
      parser.add_argument('--k-test', default=5, type=int)
      parser.add_argument('--q-train', default=15, type=int)
      parser.add_argument('--q-test', default=1, type=int)
      parser.add_argument('--lstm-layers', default=1, type=int)
      parser.add_argument('--unrolling-steps', default=2, type=int)
      parser.add_argument('--torchbench', dest='torchbench', action='store_true')
      parser.set_defaults(torchbench=False)
      args = parser.parse_args()

    self.args = args

    self.evaluation_episodes = eval_bs
    self.episodes_per_epoch = train_bs
    self.persistent_workers = True

    if args.dataset == 'omniglot':
      n_epochs = 100
      dataset_class = OmniglotDataset
      num_input_channels = 1
      lstm_input_size = 64
    elif args.dataset == 'miniImageNet':
      n_epochs = 200
      dataset_class = MiniImageNet
      num_input_channels = 3
      lstm_input_size = 1600
    else:
      raise(ValueError, 'Unsupported dataset')

    if args.torchbench:
      self.n_epochs = 3
      self.num_workers = 8
      self.persistent_workers = True

    self.param_str = f'{args.dataset}_n={args.n_train}_k={args.k_train}_q={args.q_train}_' \
                f'nv={args.n_test}_kv={args.k_test}_qv={args.q_test}_'\
                f'dist={args.distance}_fce={args.fce}'


    #########
    # Model #
    #########
    from ..few_shot.models import MatchingNetwork
    self.model = MatchingNetwork(args.n_train, args.k_train, args.q_train, args.fce, num_input_channels,
                            lstm_layers=args.lstm_layers,
                            lstm_input_size=lstm_input_size,
                            unrolling_steps=args.unrolling_steps,
                            device=self.device)
    self.model.to(self.device, dtype=torch.double)

    ###################
    # Create datasets #
    ###################
    background = dataset_class('background')
    self.background_taskloader = DataLoader(
      background,
      batch_sampler=NShotTaskSampler(background, self.episodes_per_epoch, args.n_train, args.k_train, args.q_train),
      num_workers=self.num_workers,
      persistent_workers=self.persistent_workers
    )
    evaluation = dataset_class('evaluation')
    self.evaluation_taskloader = DataLoader(
      evaluation,
      batch_sampler=NShotTaskSampler(evaluation, self.episodes_per_epoch, args.n_test, args.k_test, args.q_test),
      num_workers=self.num_workers,
      persistent_workers=self.persistent_workers
    )

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
    print(f'Training Matching Network on {self.args.dataset}...')
    optimiser = Adam(self.model.parameters(), lr=1e-3)
    loss_fn = torch.nn.NLLLoss().cuda()

    self.callbacks = [
      EvaluateFewShot(
        eval_fn=matching_net_episode,
        num_tasks=self.evaluation_episodes,
        n_shot=self.args.n_test,
        k_way=self.args.k_test,
        q_queries=self.args.q_test,
        taskloader=self.evaluation_taskloader,
        prepare_batch=prepare_nshot_task(self.args.n_test, self.args.k_test, self.args.q_test),
        fce=self.args.fce,
        distance=self.args.distance
      ),
      ModelCheckpoint(
        filepath=PATH + f'/models/matching_nets/{self.param_str}.pth',
        monitor=f'val_{self.args.n_test}-shot_{self.args.k_test}-way_acc',
        # monitor=f'val_loss',
      ),
      ReduceLROnPlateau(patience=20, factor=0.5, monitor=f'val_{self.args.n_test}-shot_{self.args.k_test}-way_acc'),
      #CSVLogger(PATH + f'/logs/matching_nets/{self.param_str}.csv'),
    ]

    fit(
      self.model,
      optimiser,
      loss_fn,
      epochs=self.n_epochs,
      dataloader=self.background_taskloader,
      prepare_batch=prepare_nshot_task(self.args.n_train, self.args.k_train, self.args.q_train),
      callbacks=self.callbacks,
      metrics=['categorical_accuracy'],
      fit_function=matching_net_episode,
      fit_function_kwargs={'n_shot': self.args.n_train, 'k_way': self.args.k_train, 'q_queries': self.args.q_train, 'train': True,
                           'fce': self.args.fce, 'distance': self.args.distance}
    )
