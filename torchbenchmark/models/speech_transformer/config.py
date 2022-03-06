import os
import json
import torch
import kaldi_io
import dataclasses

from .speech_transformer.transformer.decoder import Decoder
from .speech_transformer.transformer.encoder import Encoder
from .speech_transformer.transformer import Transformer
from .speech_transformer.transformer.optimizer import TransformerOptimizer
from .speech_transformer.transformer.loss import cal_performance
from .speech_transformer.utils import add_results_to_json, process_dict, IGNORE_ID
from .speech_transformer.data import build_LFR_features
from .speech_transformer.data import AudioDataLoader, AudioDataset

@dataclasses.dataclass
class SpeechTransformerTrainConfig:
    # Low Frame Rate
    LFR_m = 4
    LFR_n = 3
    # Network Architecture - Encoder
    d_input = 80
    n_layers_enc = 6
    n_head = 8
    d_k = 64
    d_v = 64
    d_model = 512
    d_inner = 2048
    dropout = 0.1
    pe_maxlen = 5000
    d_word_vec = 512
    n_layers_dec = 6
    tgt_emb_prj_weight_sharing = 1
    label_smoothing = 0.1
    # minibatch
    shuffle = 1
    batch_frames = 15000
    maxlen_in = 800
    maxlen_out = 150
    # don't use subprocess in dataloader
    # because TorchBench is only running 1 batch
    num_workers = 0
    # original value
    # num_workers = 4
    # optimizer
    k = 0.2
    warmup_steps = 1
    # solver configs
    epochs = 5
    save_folder = "output_data"
    checkpoint = False
    continue_from = False
    model_path = 'final.pth.tar'
    print_freq = 10
    visdom = 0
    visdom_lr = 0
    visdom_epoch = 0
    visdom_id = 0
    cross_valid = False
    # The input files. Their paths are relative to the directory of __file__
    train_json = "input_data/train/data.json"
    valid_json = "input_data/dev/data.json"
    dict_txt = "input_data/lang_1char/train_chars.txt"
    def __init__(self, prefetch=True, train_bs=32, num_train_batch=1):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.train_json = os.path.join(dir_path, self.train_json)
        self.valid_json = os.path.join(dir_path, self.valid_json)
        self.dict_txt = os.path.join(dir_path, self.dict_txt)
        self.char_list, self.sos_id, self.eos_id = process_dict(self.dict_txt)
        self.vocab_size = len(self.char_list)
        self.tr_dataset = AudioDataset(self.train_json, train_bs,
                                       self.maxlen_in, self.maxlen_out,
                                       batch_frames=self.batch_frames)
        self.cv_dataset = AudioDataset(self.valid_json, train_bs,
                                       self.maxlen_in, self.maxlen_out,
                                       batch_frames=self.batch_frames)
        self.tr_loader = AudioDataLoader(self.tr_dataset, batch_size=train_bs,
                                         num_workers=self.num_workers,
                                         shuffle=self.shuffle,
                                         LFR_m=self.LFR_m,
                                         LFR_n=self.LFR_n)
        self.cv_loader = AudioDataLoader(self.cv_dataset, batch_size=train_bs,
                                         num_workers=self.num_workers,
                                         LFR_m=self.LFR_m,
                                         LFR_n=self.LFR_n)
        self.data = {'tr_loader': self.tr_loader, 'cv_loader': self.cv_loader}
        self.encoder = Encoder(self.d_input * self.LFR_m,
                               self.n_layers_enc,
                               self.n_head,
                               self.d_k, self.d_v,
                               self.d_model, self.d_inner,
                               dropout=self.dropout, pe_maxlen=self.pe_maxlen)
        self.decoder = Decoder(self.sos_id, self.eos_id, self.vocab_size,
                               self.d_word_vec, self.n_layers_dec, self.n_head,
                               self.d_k, self.d_v, self.d_model, self.d_inner,
                               dropout=self.dropout,
                               tgt_emb_prj_weight_sharing=self.tgt_emb_prj_weight_sharing,
                               pe_maxlen=self.pe_maxlen)
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)
        self.model = Transformer(self.encoder, self.decoder)
        self.optimizer = TransformerOptimizer(torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09),
                                              self.k, self.d_model, self.warmup_steps)
        self._reset()
        self.data_loader = self.tr_loader if not SpeechTransformerTrainConfig.cross_valid else self.cv_loader
        if prefetch:
            result = []
            for _batch_num, data in zip(range(num_train_batch), self.data_loader):
                padded_input, input_lengths, padded_target = data
                padded_input = padded_input.cuda()
                input_lengths = input_lengths.cuda()
                padded_target = padded_target.cuda()
                result.append((padded_input, input_lengths, padded_target))
            self.data_loader = result

    def _reset(self):
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False

    def _run_one_epoch(self, cross_valid=False):
        total_loss = 0
        data_loader = self.data_loader
        for i, (data) in enumerate(data_loader):
            padded_input, input_lengths, padded_target = data
            padded_input = padded_input.cuda()
            input_lengths = input_lengths.cuda()
            padded_target = padded_target.cuda()
            pred, gold = self.model(padded_input, input_lengths, padded_target)
            loss, n_correct = cal_performance(pred, gold,
                                              smoothing=self.label_smoothing)
            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            non_pad_mask = gold.ne(IGNORE_ID)
            n_word = non_pad_mask.sum().item()
            return total_loss / (i + 1)

    def train(self, epoch = 1):
        self.model.train()
        tr_avg_loss = self._run_one_epoch()
        # Cross validation
        self.model.eval()
        val_loss = self._run_one_epoch(cross_valid=SpeechTransformerTrainConfig.cross_valid)
        self.tr_loss[epoch] = tr_avg_loss
        self.cv_loss[epoch] = val_loss
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

@dataclasses.dataclass
class SpeechTransformerEvalConfig:
    beam_size = 5
    nbest = 1
    decode_max_len = 100
    recog_word = 1
    # The input files. Their paths are relative to the directory of __file__
    recog_json = "input_data/test/data.json"
    dict_txt = "input_data/lang_1char/train_chars.txt"
    def __init__(self, traincfg, num_eval_batch=1):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.base_path = dir_path
        self.recog_json = os.path.join(dir_path, self.recog_json)
        self.dict_txt = os.path.join(dir_path, self.dict_txt)
        # Construct the model
        self.model, self.LFR_m, self.LFR_n = Transformer(traincfg.encoder, traincfg.decoder), traincfg.LFR_m, traincfg.LFR_n
        self.char_list, self.sos_id, self.eos_id = process_dict(self.dict_txt)
        assert self.model.decoder.sos_id == self.sos_id and self.model.decoder.eos_id == self.eos_id
        # Read json data
        with open(self.recog_json, "rb") as f:
            self.js = json.load(f)['utts']
        self.example_inputs = []
        for idx, name in enumerate(list(self.js.keys())[:self.recog_word], 1):
            feat_path = os.path.join(self.base_path, self.js[name]['input'][0]['feat'])
            input = kaldi_io.read_mat(feat_path)
            input = build_LFR_features(input, self.LFR_m, self.LFR_n)
            input = torch.from_numpy(input).float()
            input_length = torch.tensor([input.size(0)], dtype=torch.int)
            input = input.cuda()
            input_length = input_length.cuda()
            self.example_inputs.append((input, input_length))
            if len(self.example_inputs) == num_eval_batch:
                break
    def eval(self):
        with torch.no_grad():
            for input, input_length in self.example_inputs:
                nbest_hyps = self.model.recognize(input, input_length, self.char_list, self)
        return nbest_hyps
