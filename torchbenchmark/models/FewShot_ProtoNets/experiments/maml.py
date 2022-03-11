"""
Reproduce Model-agnostic Meta-learning results (supervised only) of Finn et al
"""
from torch.utils.data import DataLoader
from torch import nn
import argparse

from ..few_shot.datasets import OmniglotDataset #, MiniImageNet, DummyDataset
from ..few_shot.core import NShotTaskSampler, create_nshot_task_label, EvaluateFewShot
from ..few_shot.maml import meta_gradient_step
from ..few_shot.models import FewShotClassifier
from ..few_shot.train import fit
from ..few_shot.callbacks import *
from ..few_shot.utils import setup_dirs
from ..config import PATH

class MAML:
  def __init__(self, eval_bs = 1000, train_bs = 100, use_cuda = True, dataset='omniglot',
               n=1, k=5, q=1, inner_train_steps=1, inner_val_steps=3, inner_lr=0.4, meta_lr=0.001, meta_batch_size = 32,
               order=1, epochs=50, epoch_len=100, eval_batches=20 ):
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
      args.n=n
      args.k=k
      args.q=q
      args.inner_train_steps=inner_train_steps
      args.inner_val_steps=inner_val_steps
      args.inner_lr = inner_lr
      args.meta_lr = meta_lr
      args.meta_batch_size = meta_batch_size
      args.order = order
      args.epochs = epochs
      args.epoch_len = epoch_len
      args.eval_batches = eval_batches
    else:
      parser = argparse.ArgumentParser()
      parser.add_argument('--dataset')
      parser.add_argument('--n', default=1, type=int)
      parser.add_argument('--k', default=5, type=int)
      parser.add_argument('--q', default=1, type=int)  # Number of examples per class to calculate meta gradients with
      parser.add_argument('--inner-train-steps', default=1, type=int)
      parser.add_argument('--inner-val-steps', default=3, type=int)
      parser.add_argument('--inner-lr', default=0.4, type=float)
      parser.add_argument('--meta-lr', default=0.001, type=float)
      parser.add_argument('--meta-batch-size', default=32, type=int)
      parser.add_argument('--order', default=1, type=int)
      parser.add_argument('--epochs', default=50, type=int)
      parser.add_argument('--epoch-len', default=100, type=int)
      parser.add_argument('--eval-batches', default=20, type=int)
      parser.add_argument('--torchbench', dest='torchbench', action='store_true')
      parser.set_defaults(torchbench=False)

      args = parser.parse_args()

    self.args = args

    if self.args.dataset == 'omniglot':
      self.dataset_class = OmniglotDataset
      self.fc_layer_size = 64
      self.num_input_channels = 1
    elif self.args.dataset == 'miniImageNet':
      self.dataset_class = MiniImageNet
      self.fc_layer_size = 1600
      self.num_input_channels = 3
    elif self.args.dataset == 'dummy':
      self.dataset_class = DummyDataset
      self.fc_layer_size = 64
      self.num_input_channels = 1
    else:
      raise(ValueError('Unsupported dataset'))

    #print('inner train steps ', args.inner_train_steps, ' inner-val-steps ', args.inner_val_steps)

    self.param_str = f'{self.args.dataset}_order={self.args.order}_n={self.args.n}_k={self.args.k}_metabatch={self.args.meta_batch_size}_' \
                f'train_steps={self.args.inner_train_steps}_val_steps={self.args.inner_val_steps}'
    #print(param_str)

    if self.args.torchbench:
      self.num_workers = 8
      self.pin_memory = False
      self.prefetch_factor = 2
      self.persistent_workers = True
      self.args.epochs = 3
    else:
      self.num_workers = 8
      self.pin_memory = False
      self.prefetch_factor = 2
      self.persistent_workers = True

    ###################
    # Create datasets #
    ###################
    background = self.dataset_class('background')
    self.background_taskloader = DataLoader(
      background,
      batch_sampler=NShotTaskSampler(background, self.args.epoch_len, n=self.args.n, k=self.args.k, q=self.args.q,
                                     num_tasks=self.args.meta_batch_size),
      num_workers=self.num_workers,
      pin_memory=self.pin_memory,
      prefetch_factor = self.prefetch_factor,
      persistent_workers = self.persistent_workers
    )
    evaluation = self.dataset_class('evaluation')
    self.evaluation_taskloader = DataLoader(
      evaluation,
      batch_sampler=NShotTaskSampler(evaluation, self.args.eval_batches, n=self.args.n, k=self.args.k, q=self.args.q,
                                     num_tasks=self.args.meta_batch_size),
      num_workers=self.num_workers,
      pin_memory=self.pin_memory,
      prefetch_factor = self.prefetch_factor,
      persistent_workers = self.persistent_workers
    )

    # training init
    self.meta_model = FewShotClassifier(self.num_input_channels, self.args.k, self.fc_layer_size).to(self.device, dtype=torch.double)
    self.meta_optimiser = torch.optim.Adam(self.meta_model.parameters(), lr=self.args.meta_lr)
    self.loss_fn = nn.CrossEntropyLoss().to(self.device)

    self.eval_fake = torch.randn(self.eval_bs, 1, 28, 28, device=self.device, dtype=torch.double)

  def get_module(self):
    return self.meta_model, self.eval_fake

  def Eval(self, niter=200):
    self.meta_model.eval()
    for _ in range(niter):
      x = self.meta_model(self.eval_fake).view(-1)

    return

  def Train(self):


    def prepare_meta_batch(device, n, k, q, meta_batch_size, num_input_channels):

      def prepare_meta_batch_(batch):
        x, y = batch
        # Reshape to `meta_batch_size` number of tasks. Each task contains
        # n*k support samples to train the fast model on and q*k query samples to
        # evaluate the fast model on and generate meta-gradients
        x = x.reshape(meta_batch_size, n*k + q*k, num_input_channels, x.shape[-2], x.shape[-1])
        # Move to device

        #print('shape:', x.shape)

        x = x.double().to(device)
        # Create label
        y = create_nshot_task_label(k, q).cuda().repeat(meta_batch_size)
        return x, y

      return prepare_meta_batch_

    ############
    # Training #
    ############
    print(f'Training MAML on {self.args.dataset}...')

    self.callbacks = [
      EvaluateFewShot(
        eval_fn=meta_gradient_step,
        num_tasks=self.args.eval_batches,
        n_shot=self.args.n,
        k_way=self.args.k,
        q_queries=self.args.q,
        taskloader=self.evaluation_taskloader,
        prepare_batch=prepare_meta_batch(self.device, self.args.n, self.args.k, self.args.q, self.args.meta_batch_size, self.num_input_channels),
        # MAML kwargs
        inner_train_steps=self.args.inner_val_steps,
        inner_lr=self.args.inner_lr,
        device=self.device,
        order=self.args.order,
      ),
      ModelCheckpoint(
        filepath=PATH + f'/models/maml/{self.param_str}.pth',
        monitor=f'val_{self.args.n}-shot_{self.args.k}-way_acc'
      ),
      ReduceLROnPlateau(patience=10, factor=0.5, monitor=f'val_loss'),
      CSVLogger(PATH + f'/logs/maml/{self.param_str}.csv'),
    ]

    fit(
      self.meta_model,
      self.meta_optimiser,
      self.loss_fn,
      epochs=self.args.epochs,
      dataloader=self.background_taskloader,
      prepare_batch=prepare_meta_batch(self.device, self.args.n, self.args.k, self.args.q, self.args.meta_batch_size, self.num_input_channels),
      callbacks=self.callbacks,
      metrics=['categorical_accuracy'],
      fit_function=meta_gradient_step,
      fit_function_kwargs={'n_shot': self.args.n, 'k_way': self.args.k, 'q_queries': self.args.q,
                           'train': True,
                           'order': self.args.order, 'device': self.device, 'inner_train_steps': self.args.inner_train_steps,
                           'inner_lr': self.args.inner_lr},
    )
