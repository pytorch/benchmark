from .trainer import Trainer
from torch.nn.parallel import DistributedDataParallel as DDP
from torchbenchmark.util.e2emodel import E2EBenchmarkModel
import torch.distributed as dist

class DeepSpeedTrainer(Trainer):
    DEFAULT_MEASURE_ITERATIONS = 10
    def __init__(self, args, model_class, batch_size=None, extra_args=[]):
        super().__init__(args, model_class, mode="SPMD")
        
        self.setup()
        # create model instance after Trainer setup, so that
        # visible devices won't be revised in model constructor

        #TODO(whc) use_deepspeed breaks the API... but it is most convenient to initialize/configure deepspeed
        # inside the model constructur rather than later, since later is too late to intercept the Accelerator
        # configuration for hf_ models
        model: E2EBenchmarkModel = model_class("train", batch_size, use_deepspeed=True, extra_args=extra_args)

        expected_attrs = ["model", "optimizer", "train_dataloader", "accelerator"]
        assert all(attr in dir(model) for attr in expected_attrs), (
            "Missing attributes in the input E2EBenchmarkModel implementation: "
            f"{[attr for attr in expected_attrs if attr not in dir(model)]}"
        )

        self.model = model.model
        self.optimizer = model.optimizer
        self.dataloader = model.train_dataloader
        self.accelerator = model.accelerator

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
    
    def next_batch(self):
        return next(iter(self.dataloader))

    def forward(self, input):
        """
        compute model forward and return loss
        """
        return self.model(**input).loss
    
    def backward(self, loss):
        self.accelerator.backward(loss)

    def optimizer_step(self):
        self.optimizer.step()

def test():
    from torchbenchmark.e2e_models.hf_bert import Model

    import os

    os.environ["LOCAL_RANK"] = str(0)
    os.environ["WORLD_SIZE"] = str(1)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "5678"
    os.environ["RANK"] = str(0)

    trainer = DeepSpeedTrainer(Model)

    trainer.measure()

if __name__=="__main__":
    test()

