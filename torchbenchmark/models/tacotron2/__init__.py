from .train_tacotron2 import train, load_model, prepare_dataloaders
import torch
from .loss_function import Tacotron2Loss
from argparse import Namespace
from .text import symbols
from pathlib import Path

class Model:
    def __init__(self, device='cpu', jit=False):
        """ Required """
        self.device = device
        self.jit = jit
        if device == 'cpu' or jit:
            # TODO - currently load_model assumes cuda
            return

        self.hparams = self.create_hparams()
        self.model = load_model(self.hparams).to(device=device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate,
                                 weight_decay=self.hparams.weight_decay)
        self.criterion = Tacotron2Loss().to(device=device)
        train_loader, valset, collate_fn = prepare_dataloaders(self.hparams)
        self.example_input, self.target = self.model.parse_batch(list(train_loader)[0], device=self.device)


    def create_hparams(hparams_string=None, verbose=False):
        """Create model hyperparameters. Parse nondefault from given string."""
        root = str(Path(__file__).parent)
        hparams = Namespace(**{
            ################################
            # Experiment Parameters        #
            ################################
            'epochs':2,
            'iters_per_checkpoint':1000,
            'seed':1234,
            'dynamic_loss_scaling':True,
            'fp16_run':False,
            'distributed_run':False,
            'dist_backend':"nccl",
            'dist_url':"tcp://localhost:54321",
            'cudnn_enabled':True,
            'cudnn_benchmark':False,
            'ignore_layers':['embedding.weight'],

            ################################
            # Data Parameters             #
            ################################
            'load_mel_from_disk':False,
            'training_files':f'{root}/filelists/ljs_audio_text_train_filelist.txt',
            'validation_files':f'{root}/filelists/ljs_audio_text_val_filelist.txt',
            'text_cleaners':['english_cleaners'],

            ################################
            # Audio Parameters             #
            ################################
            'max_wav_value':32768.0,
            'sampling_rate':22050,
            'filter_length':1024,
            'hop_length':256,
            'win_length':1024,
            'n_mel_channels':80,
            'mel_fmin':0.0,
            'mel_fmax':8000.0,

            ################################
            # Model Parameters             #
            ################################
            'n_symbols':len(symbols),
            'symbols_embedding_dim':512,

            # Encoder parameters
            'encoder_kernel_size':5,
            'encoder_n_convolutions':3,
            'encoder_embedding_dim':512,

            # Decoder parameters
            'n_frames_per_step':1,  # currently only 1 is supported
            'decoder_rnn_dim':1024,
            'prenet_dim':256,
            'max_decoder_steps':1000,
            'gate_threshold':0.5,
            'p_attention_dropout':0.1,
            'p_decoder_dropout':0.1,

            # Attention parameters
            'attention_rnn_dim':1024,
            'attention_dim':128,

            # Location Layer parameters
            'attention_location_n_filters':32,
            'attention_location_kernel_size':31,

            # Mel-post processing network parameters
            'postnet_embedding_dim':512,
            'postnet_kernel_size':5,
            'postnet_n_convolutions':5,

            ################################
            # Optimization Hyperparameters #
            ################################
            'use_saved_learning_rate':False,
            'learning_rate':1e-3,
            'weight_decay':1e-6,
            'grad_clip_thresh':1.0,
            'batch_size':1,
            'mask_padding':True  # set model's padded outputs to padded values
        })
        return hparams


    def get_module(self):
        if self.device == 'cpu':
            raise NotImplementedError('CPU not supported')
        if self.jit:
            raise NotImplementedError('JIT not supported')
        return self.model, (self.example_input,)

    def train(self, niterations=1):
        if self.device == 'cpu':
            raise NotImplementedError("Disabled due to excessively slow runtime - see GH Issue #100")
        if self.jit:
            raise NotImplementedError('JIT not supported')
        self.model.train()
        for _ in range(niterations):
            self.model.zero_grad()
            y_pred = self.model(self.example_input)

            loss = self.criterion(y_pred, self.target)
            loss.backward()
            self.optimizer.step()


    def eval(self, niterations=1):
        if self.device == 'cpu':
            raise NotImplementedError('CPU not supported')
        if self.jit:
            raise NotImplementedError('JIT not supported')
        self.model.eval()
        for _ in range(niterations):
            self.model(self.example_input)


if __name__ == '__main__':
    m = Model(device='cuda', jit=False)
    model = m.get_module()
    model, example_inputs = m.get_module()
    model(*example_inputs)
    m.train()
    m.eval()
