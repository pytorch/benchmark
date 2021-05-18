import json
import torch
import kaldi_io
import dataclasses

from decoder import Decoder
from encoder import Encoder
from solver import Solver
from transformer import Transformer
from transformer.optimizer import TransformerOptimizer
from utils import add_results_to_json, process_dict
from data import build_LFR_features
from data import AudioDataLoader, AudioDataset

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
    batch_size = 16
    batch_frames = 15000
    maxlen_in = 800
    maxlen_out = 150
    num_workers = 4
    # optimizer
    k = 0.2
    warmup_steps = 4000
    # input files
    train_json = "input_data/train/data.json"
    valid_json = "input_data/dev/data.json"
    dict_txt = "input_data/lang_1char/train_chars.txt"
    def __init__(self):
        self.tr_dataset = AudioDataset(self.train_json, self.cfg.batch_size,
                                       self.cfg.maxlen_in, self.cfg.maxlen_out,
                                       batch_frames=self.batch_frames)
        self.cv_dataset = AudioDataset(self.valid_json, self.cfg.batch_size,
                                       self.cfg.maxlen_in, self.cfg.maxlen_out,
                                       batch_frames=self.cfg.batch_frames)
        self.tr_loader = AudioDataLoader(self.tr_dataset, batch_size=1,
                                         num_workers=self.cfg.num_workers,
                                         shuffle=self.cfg.shuffle,
                                         LFR_m=self.cfg.LFR_m,
                                         LFR_n=self.cfg.LFR_n)
        self.cv_loader = AudioDataLoader(self.cv_dataset, batch_size=1,
                                         num_workers=self.cfg.num_workers,
                                         LFR_m=self.cfg.LFR_m,
                                         LFR_n=self.cfg.LFR_n)
        self.encoder = Encoder(self.cfg.d_input * self.cfg.LFR_m,
                               self.cfg.n_layers_enc,
                               self.cfg.n_head,
                               self.cfg.d_k, self.cfg.d_v,
                               self.cfg.d_model, self.cfg.d_inner,
                               dropout=self.cfg.dropout, pe_maxlen=self.cfg.pe_maxlen)
        self.decoder = Decoder(self.sos_id, self.eos_id, self.vocab_size,
                               self.cfg.d_word_vec, self.cfg.n_layers_dec, self.cfg.n_head,
                               self.cfg.d_k, self.cfg.d_v, self.cfg.d_model, self.cfg.d_inner,
                               dropout=self.cfg.dropout,
                               tgt_emb_prj_weight_sharing=self.cfg.tgt_emb_prj_weight_sharing,
                               pe_maxlen=self.cfg.pe_maxlen)
        self.model = Transformer(self.encoder, self.decoder)
        self.optimizer = TransformerOptimizer(torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
                                              self.k, self.d_model, self.warmup_steps)
        self.solver = Solver(self.data, self.model, self.optimizer, self)
    def train(self):
        self.solver.train()

@dataclasses.dataclass
class SpeechTransformerEvalConfig:
    beam_size = 5
    nbest = 1
    decode_max_len = 100
    recog_json = "input_data/test/data.json"
    dict_txt = "input_data/data/lang_1char/train_chars.txt"
    def __init__(self, traincfg):
        # Construct the model
        self.model, self.LFR_m, self.LFR_n = Transformer(trancfg.encoder, trancfg.decoder), traincfg.LFR_m, traincfg.LFR_n
        self.char_list, self.sos_id, self.eos_id = process_dict(self.dict_txt)
        assert model.decoder.sos_id == sos_id and model.decoder.eos_id == eos_id
        # Read json data
        with open(self.recog_json, "rb") as f:
            self.js = json.load(f)['utts']
    def eval(self):
        with torch.no_grad():
            for idx, name in enumerate(self.js.keys(), 1):
                input = kaldi_io.read_mat(self.js[name]['input'][0]['feat'])
                input = build_LFR_features(input, self.LFR_m, self.LFR_n)
                input = torch.from_numpy(input).float()
                input_length = torch.tensor([input.size(0)], dtype=torch.int)
                input = input.cuda()
                input_length = input_length.cuda()
                nbest_hyps = self.model.recognize(input, input_length, self.char_list, self)
