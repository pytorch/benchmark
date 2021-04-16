#!/usr/bin/env python

import torch

from .data import AudioDataLoader, AudioDataset
from .decoder import Decoder
from .encoder import Encoder
from .transformer import Transformer
from .solver import Solver
from .optimizer import TransformerOptimizer
from .config import SpeechTransformerConfig, SpeechTransformerEvalConfig

from torchbenchmark.tasks import SPEECH

class Model(BenchmarkModel, device = "cuda", jit = False):
    task = SPEECH.RECOGNITION
    def __init__(self, device=None, jit=False):
        self.train_json = None
        self.valid_json = None
        self.dict = None
        self.cfg = SpeechTransformerConfig()
        self.evalcfg = SpeechTransformerEvalConfig()
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
        self.char_list, self.sos_id, self.eos_id = process_dict(self.dict)
        self.vocab_size = len(self.char_list)
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
        model = Transformer(encoder, decoder)
        if self.device == "cuda":
            model.cuda()
        self.optimizer = TransformerOptimizer(torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
                                              self.cfg.k, self.cfg.d_model, self.cfg.warmup_steps)
        self.solver = Solver(data, model, optimizier, self.cfg)
        # Read eval json data
        with open(self.recog_json, 'rb') as f:
            self.js = json.load(self.recog_json)['utts']

    def get_module(self):
        pass

    def train(self, niter=1):
        for _ in range(niter):
            self.solver.train()

    def eval(self, niter=1):
        self.model.eval()
        with torch.no_grad():
            for _ in range(niter):
                for idx, name in enumerate(self.js.keys(), 1):
                    nbest_hyps = model.recognize(self.evalcfg.input,
                                                 self.evalcfg.input_length,
                                                 self.evalcfg.char_list)
                    new_js[name] = add_results_to_json(js[name], nbest_hyps, self.char_list)

if __name__ == '__main__':
    for device in ['cuda']:
        for jit in [False]:
            m = Model(device=device, jit=jit)
            model, example_inputs = m.get_module()
            model(*example_inputs)
            m.train()
            m.eval()
