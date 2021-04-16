import dataclasses


@dataclasses.dataclass
class SpeechTransformerConfig:
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
    shuffle = 0
    batch_size = 32
    batch_frames = 0
    maxlen_in = 800
    maxlen_out = 150
    num_workers = 4
    # optimizer
    k = 1.0
    warmup_steps = 4000

@dataclasses.dataclass
class SpeechTransformerEvalConfig:
    beam_size = 1
    nbest = 1
    decode_max_len = 0
    def __init__(self):
        # Construct self.input
