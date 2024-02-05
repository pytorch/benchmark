from torchbenchmark.util.framework.gnn.model_factory import BasicGNNModel


class Model(BasicGNNModel):
    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(
            model_name="edgecnn",
            test=test,
            device=device,
            batch_size=batch_size,
            extra_args=extra_args,
        )
