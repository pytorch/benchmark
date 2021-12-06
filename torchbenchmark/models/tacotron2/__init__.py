from .train_tacotron2 import load_model, prepare_dataloaders
import torch
from .loss_function import Tacotron2Loss
from argparse import Namespace
from .text import symbols
from pathlib import Path
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import SPEECH


class Model(BenchmarkModel):
    task = SPEECH.SYNTHESIS

    # Training batch size comes from the source code:
    # Source: https://github.com/NVIDIA/tacotron2/blob/bb6761349354ee914909a42208e4820929612069/hparams.py#L84
    def __init__(self, device=None, jit=False, train_bs=64, eval_bs=64):
        super().__init__()
        """ Required """
        self.device = device
        self.jit = jit
        if device == 'cpu' or jit:
            # TODO - currently load_model assumes cuda
            return

        self.train_hparams = self.create_hparams(batch_size=train_bs)
        self.eval_hparams = self.create_hparams(batch_size=eval_bs)
        self.train_model = load_model(self.train_hparams).to(device=device)
        self.eval_model = load_model(self.eval_hparams).to(device=device)
        self.train_optimizer = torch.optim.Adam(self.train_model.parameters(),
                                                lr=self.train_hparams.learning_rate,
                                                weight_decay=self.train_hparams.weight_decay)
        self.eval_optimizer = torch.optim.Adam(self.eval_model.parameters(),
                                               lr=self.eval_hparams.learning_rate,
                                               weight_decay=self.eval_hparams.weight_decay)
        self.criterion = Tacotron2Loss().to(device=device)
        train_loader, valset, collate_fn = prepare_dataloaders(self.train_hparams)
        self.example_input, self.target = self.train_model.parse_batch(next(iter(train_loader)), device=self.device)

    # Parameters were obtained from the source code.
    # Source: https://github.com/NVIDIA/tacotron2/blob/bb6761349354ee914909a42208e4820929612069/hparams.py#L5
    def create_hparams(hparams_string=None, verbose=False, batch_size=64):
        """Create model hyperparameters. Parse nondefault from given string."""
        root = str(Path(__file__).parent)
        hparams = Namespace(**{
            ################################
            # Experiment Parameters        #
            ################################
            'epochs': 2,  # Reduced in TorchBench to shorten number of train iterations.
            'iters_per_checkpoint': 1000,
            'seed': 1234,
            'dynamic_loss_scaling': True,
            'fp16_run': False,
            'distributed_run': False,
            'dist_backend': "nccl",
            'dist_url': "tcp://localhost:54321",
            'cudnn_enabled': True,
            'cudnn_benchmark': False,
            'ignore_layers': ['embedding.weight'],

            ################################
            # Data Parameters             #
            ################################
            'load_mel_from_disk': False,
            'training_files': f'{root}/filelists/ljs_audio_text_train_filelist.txt',
            'validation_files': f'{root}/filelists/ljs_audio_text_val_filelist.txt',
            'text_cleaners': ['english_cleaners'],

            ################################
            # Audio Parameters             #
            ################################
            'max_wav_value': 32768.0,
            'sampling_rate': 22050,
            'filter_length': 1024,
            'hop_length': 256,
            'win_length': 1024,
            'n_mel_channels': 80,
            'mel_fmin': 0.0,
            'mel_fmax': 8000.0,

            ################################
            # Model Parameters             #
            ################################
            'n_symbols': len(symbols),
            'symbols_embedding_dim': 512,

            # Encoder parameters
            'encoder_kernel_size': 5,
            'encoder_n_convolutions': 3,
            'encoder_embedding_dim': 512,

            # Decoder parameters
            'n_frames_per_step': 1,  # currently only 1 is supported
            'decoder_rnn_dim': 1024,
            'prenet_dim': 256,
            'max_decoder_steps': 1000,
            'gate_threshold': 0.5,
            'p_attention_dropout': 0.1,
            'p_decoder_dropout': 0.1,

            # Attention parameters
            'attention_rnn_dim': 1024,
            'attention_dim': 128,

            # Location Layer parameters
            'attention_location_n_filters': 32,
            'attention_location_kernel_size': 31,

            # Mel-post processing network parameters
            'postnet_embedding_dim': 512,
            'postnet_kernel_size': 5,
            'postnet_n_convolutions': 5,

            ################################
            # Optimization Hyperparameters #
            ################################
            'use_saved_learning_rate': False,
            'learning_rate': 1e-3,
            'weight_decay': 1e-6,
            'grad_clip_thresh': 1.0,
            'batch_size': batch_size,
            'mask_padding': True  # set model's padded outputs to padded values
        })
        return hparams

    def get_module(self):
        if self.device == 'cuda':
            raise NotImplementedError('CUDA disabled due to CUDA out of memory on CI GPU')
        if self.device == 'cpu':
            raise NotImplementedError('CPU not supported')
        if self.jit:
            raise NotImplementedError('JIT not supported')
        return self.train_model, (self.example_input,)

    def train(self, niterations=1):
        if self.device == 'cuda':
            raise NotImplementedError('CUDA disabled due to CUDA out of memory on CI GPU')
        if self.device == 'cpu':
            raise NotImplementedError("Disabled due to excessively slow runtime - see GH Issue #100")
        if self.jit:
            raise NotImplementedError('JIT not supported')
        self.train_model.train()
        for _ in range(niterations):
            self.train_model.zero_grad()
            y_pred = self.train_model(self.example_input)

            loss = self.criterion(y_pred, self.target)
            loss.backward()
            self.train_optimizer.step()

    def eval(self, niterations=1):
        if self.device == 'cuda':
            raise NotImplementedError('CUDA disabled due to CUDA out of memory on CI GPU')
        if self.device == 'cpu':
            raise NotImplementedError('CPU not supported')
        if self.jit:
            raise NotImplementedError('JIT not supported')
        self.eval_model.eval()
        for _ in range(niterations):
            self.eval_model(self.example_input)


if __name__ == '__main__':
    m = Model(device='cuda', jit=False)
    model, example_inputs = m.get_module()
    model(*example_inputs)
    m.train()
    m.eval()
