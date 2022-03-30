import torch
import sys
import os
from torchbenchmark import REPO_PATH
from typing import Tuple

# Import FAMBench model path
class add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass
XLMR_PATH = os.path.join(REPO_PATH, "submodules", "FAMBench", "benchmarks", "xlmr", "ootb")
with add_path(XLMR_PATH):
    from xlmr import generate_dataset
    from xlmr_parser import init_argparse

from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.tasks import NLP
import torch.nn.functional as F

class Model(BenchmarkModel):
    task = NLP.LANGUAGE_MODELING
    FAMBENCH_MODEL = True
    # original parameters: FAMBench/benchmarks/run_xlmr_ootb.sh
    DEFAULT_TRAIN_BSIZE = 16
    DEFAULT_EVAL_BSIZE = 16
    DEFAULT_SEQ_LENGTH = 16
    DEFAULT_VOCAB_SIZE = 250000
    # TorchBench only runs 1 batch
    DEFAULT_NUM_BATCHES = 1

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        self.xlmr = torch.hub.load('pytorch/fairseq:main', 'xlmr.large')
        parser = init_argparse()
        args = parser.parse_args([f"--num-batches={self.DEFAULT_NUM_BATCHES}", f"--batch-size={self.batch_size} ", \
                                 f"--sequence-length={self.DEFAULT_SEQ_LENGTH}", f"--vocab-size={self.DEFAULT_VOCAB_SIZE}"])
        if test == "train":
            self.learning_rate = 0.01
            self.optimizer = torch.optim.SGD(self.xlmr.parameters(), lr=self.learning_rate)
            self.xlmr.train()
            args.inference_only = False
        elif test == "eval":
            self.xlmr.eval()
            args.inference_only = True
        # Generate data! y is empty if inference_only.
        self.x_l, self.y_true_l = generate_dataset(args.num_batches, args.batch_size,
            args.vocab_size, args.inference_only, uniform_seqlen=args.sequence_length,
            seqlen_dist=args.seqlen_dist, seq_len_dist_max=args.seqlen_dist_max)
        # Prefetch the model and data to device
        self.xlmr = self.xlmr.to(self.device)
        self.x_l = list(map(lambda x: x.to(self.device), self.x_l))
        self.y_true_l = list(map(lambda x: x.to(self.device), self.y_true_l))

    def get_module(self):
        return self.xlmr, self.x_l

    def enable_fp16_half(self):
        self.xmlr = self.xlmr.half()

    def train(self):
        for i, (x, y_true) in enumerate(zip(self.x_l, self.y_true_l)):
            y_pred = self.xlmr.extract_features(x)
            loss = F.cross_entropy(y_pred, y_true) 
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad() 

    def eval(self) -> Tuple[torch.Tensor]:
        result = None
        with torch.no_grad():
            for i, x in enumerate(self.x_l):
                y_pred = self.xlmr.extract_features(x)
                result = y_pred
        return (result, )