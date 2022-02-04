from torchbenchmark.util.framework.vision.model_factory import TorchVisionModel
from torchbenchmark.tasks import COMPUTER_VISION

class Model(TorchVisionModel):
    task = COMPUTER_VISION.CLASSIFICATION

    # Original train batch size: 512, out of memory on V100 GPU
    # Use hierarchical batching to scale down: 512 = batch_size (32) * epoch_size (16)
    # Source: https://github.com/forresti/SqueezeNet
    def __init__(self, test, device, jit=False, train_bs=32, eval_bs=16, extra_args=[]):
        super().__init__(model_name="squeezenet1_1", test=test, device=device, jit=jit,
                         train_bs=train_bs, eval_bs=eval_bs, extra_args=extra_args)
        self.epoch_size = 16
    
    # Temporarily disable training because this will cause CUDA OOM in CI
    # TODO: re-enable this test when better hardware is available
    def train(self, niter=1):
        raise NotImplementedError("Temporarily disable training test because it causes CUDA OOM on T4")
        optimizer = optim.Adam(self.model.parameters())
        loss = torch.nn.CrossEntropyLoss()
        for _ in range(niter):
            optimizer.zero_grad()
            for _ in range(self.epoch_size):
                pred = self.model(*self.example_inputs)
                y = torch.empty(pred.shape[0], dtype=torch.long, device=self.device).random_(pred.shape[1])
                loss(pred, y).backward()
            optimizer.step()
