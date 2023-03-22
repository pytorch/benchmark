import torch
import typing
from contextlib import nullcontext
from torchbenchmark.util.model import BenchmarkModel

from torch_geometric.nn import GAT, GCN, GraphSAGE
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

    def __init__(self, model_name, test, device, jit=False, batch_size = None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        self.tb_args, self.extra_args = parse_tb_args(self.extra_args)

        root = str(Path(__file__).parent.parent.parent.parent)
        sparse = True if self.tb_args.graph_type == "sparse" else False
        if sparse:
            data = torch.load(f'{root}/data/.data/Reddit_minimal/sub_reddit_sparse.pt')
        else:
            data = torch.load(f'{root}/data/.data/Reddit_minimal/sub_reddit.pt')
        print(data)
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

        if test == "train":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            self.model.train()
        elif test == "eval":
            self.model.eval()
        self.amp_context = nullcontext

    def get_module(self):
        return self.model, self.example_inputs[0]

    def train(self):
        for batch_id in range(self.num_batch):
            self.optimizer.zero_grad()
            out = self.model(**self.example_inputs[batch_id])
            loss = F.cross_entropy(out, self.example_outputs[batch_id])
            loss.backward()
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

    def enable_amp(self):
        if not self.dynamo and self.opt_args.backend == 'cudagraph':
            return NotImplementedError("AMP not implemented for cudagraphs")
        self.amp_context = lambda: torch.cuda.amp.autocast(dtype=torch.float16)
