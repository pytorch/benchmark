from torchbenchmark.util.framework.gnn.model_factory import BasicGNNModel
from torchbenchmark.tasks import GNN

class Model(BasicGNNModel):
    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(model_name="edgecnn", test=test, device=device, jit=jit,
                         batch_size=batch_size, extra_args=extra_args)
