import torch
import sys
import typing
from contextlib import nullcontext
import copy
from torchbenchmark.util.model import BenchmarkModel

import torch_geometric
from torch_geometric.nn import GAT, GCN, GraphSAGE, GIN, EdgeCNN
from torchbenchmark.tasks import GNN
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from torch_geometric.loader import NeighborLoader
from torchbenchmark.util.framework.gnn.config import parse_tb_args
from typing import List
from torch import Tensor

models_dict = {
    'gat': GAT,
    'gcn': GCN,
    'edgecnn': EdgeCNN,
    'gin': GIN,
    'sage': GraphSAGE,
}

class GNNModel(BenchmarkModel):
    # To recognize this is a GNN model
    GNN_MODEL = True
    # These two variables should be defined by subclasses
    DEFAULT_TRAIN_BSIZE = None
    DEFAULT_EVAL_BSIZE = None
    # Default eval precision on CUDA device is fp16
    DEFAULT_EVAL_CUDA_PRECISION = "fp16"

    def __init__(self, model_name, test, device, batch_size = None, extra_args=[]):
        super().__init__(test=test, device=device, batch_size=batch_size, extra_args=extra_args)
        self.tb_args, self.extra_args = parse_tb_args(self.extra_args)

        root = str(Path(__file__).parent.parent.parent.parent)
        sparse = True if self.tb_args.graph_type == "sparse" else False
        if sparse:
            data = torch.load(f'{root}/data/.data/Reddit_minimal/sub_reddit_sparse.pt')
        else:
            data = torch.load(f'{root}/data/.data/Reddit_minimal/sub_reddit.pt')
        mask = None
        sampler = None
        kwargs = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': 0,
        }
        self.subgraph_loader = NeighborLoader(
            data,
            num_neighbors=[-1],  # layer-wise inference
            input_nodes=mask,
            sampler=sampler,
            **kwargs,
        )

        Model = models_dict.get(model_name, None)
        num_layers = 1
        hidden_channels = 64
        input_channels = data.num_features
        out_channels = 41 # num_classes
        if model_name == "gat":
            num_heads = 2
            self.model = Model(input_channels, hidden_channels, num_layers, out_channels, heads=num_heads)
        else:
            self.model = Model(input_channels, hidden_channels, num_layers, out_channels)
        self.model = self.model.to(device)
        tmp_example_inputs = []
        tmp_example_outputs = []
        self.num_batch = 0
        for batch in self.subgraph_loader:
            self.num_batch += 1
            if hasattr(batch, 'adj_t'):
                edge_index = batch.adj_t.to(device)
            else:
                edge_index = batch.edge_index.to(device)
            tmp_example_inputs.append({"x": batch.x.to(device), "edge_index": edge_index})
            tmp_example_outputs.append(batch.y.to(device))
        self.example_inputs = tmp_example_inputs
        self.example_outputs = tmp_example_outputs
        self.starter_inputs = copy.copy(self.example_inputs) # important to rerun the input generator

        if test == "train":
            self.output_generator = self._gen_target()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            self.loss_fn = torch.nn.CrossEntropyLoss()
            self.model.train()
        elif test == "eval":
            self.model.eval()
        self.amp_context = nullcontext

    def get_module(self):
        return self.model, self.example_inputs[0]

    def get_input_iter(self):
        """Yield batch of inputs."""

        while True:
            for example_input in self.starter_inputs:
                yield example_input

    def _gen_target(self):
        """Yield batch of targets"""

        while True:
            for example_output in self.example_outputs:
                yield example_output


    def forward(self):
        pred = self.model(**self.example_inputs)
        outputs = next(self.output_generator)

        return self.loss_fn(pred, outputs)


    def backward(self, loss):
        loss.backward()


    def optimizer_step(self):
        self.optimizer.step()


    def eval(self) -> typing.Tuple[torch.Tensor]:
        with self.amp_context():
            xs: List[Tensor] = []
            result = self.subgraph_loader.data.x.cpu()
            for batch_id in range(self.num_batch):
                x = self.model(**self.example_inputs[batch_id])
                xs.append(x.cpu())
            result = torch.cat(xs, dim=0)
        return (result, )

# Variation of GNNModel based off of test/nn/models/test_basic_gnn.py; the
# difference is we don't bother with data loading or optimizer step
class BasicGNNModel(BenchmarkModel):
    # This benchmark doesn't seem to have any batch size
    ALLOW_CUSTOMIZE_BSIZE = False
    DEFAULT_TRAIN_BSIZE = 1
    DEFAULT_EVAL_BSIZE = 1
    task = GNN.CLASSIFICATION
    def __init__(self, model_name, test, device, batch_size = None, extra_args=[]):
        super().__init__(test=test, device=device, batch_size=batch_size, extra_args=extra_args)
        Model = models_dict[model_name]
        self.model = Model(64, 64, num_layers=3).to(device)
        # Apply some global side effects to library (throw out the compiled
        # model though, we don't need it yet)
        torch_geometric.compile(self.model)
        # Make the model jittable
        # (TODO: This probably makes us overstate the speedup, as making the
        # model jittable also probably reduces its performance; but this is
        # matching the benchmark)
        self.model = sys.modules["torch_geometric.compile"].to_jittable(self.model)
        num_nodes, num_edges = 10_000, 200_000
        x = torch.randn(num_nodes, 64, device=device)
        edge_index = torch.randint(num_nodes, (2, num_edges), device=device)
        self.example_inputs = (x, edge_index)

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            return (self.model(*self.example_inputs),)

    def train(self):
        # NB: This is a little different than test_basic_gnn.py, as we
        # are including the cost of randn_like in the overall computation here
        out = self.model(*self.example_inputs)
        out_grad = torch.randn_like(out)
        out.backward(out_grad)

    def get_module(self):
        return self.model, self.example_inputs
