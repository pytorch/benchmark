from torchbenchmark.util.framework.gnn.model_factory import GNNModel
from torchbenchmark.tasks import GNN

class Model(GNNModel):
    task = GNN.CLASSIFICATION
    DEFAULT_TRAIN_BSIZE = 64
    DEFAULT_EVAL_BSIZE = 64

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(model_name="gcn", test=test, device=device, jit=jit,
                         batch_size=batch_size, extra_args=extra_args)
        if device == 'cuda':
            # TODO - Add CUDA support
            raise NotImplementedError("GCN doesn't support CUDA")
