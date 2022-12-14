# upstream repo: https://github.com/kuangliu/pytorch-cifar
import torch
import torchvision
import torchvision.transforms as transforms
from torchbenchmark.util.e2emodel import E2EBenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION
import os

from pathlib import Path

# setup environment variable
CURRENT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))

class Model(E2EBenchmarkModel):
    task = COMPUTER_VISION.CLASSIFICATION
    DEFAULT_TRAIN_BSIZE: int = 128
    DEFAULT_EVAL_BSIZE: int = 1

    def __init__(self, test, batch_size=None, extra_args=[]):
        super().__init__(test=test, batch_size=batch_size, extra_args=extra_args)
        self.device = "cuda"
        data_root = CURRENT_DIR.joinpath(".data")
        assert torch.cuda.is_available(), f"This model requires CUDA device available."
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(
            root=str(data_root), train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root=str(data_root), train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')
        self.lr = 0.1
        # initial accuracy
        self.acc = 0

        if self.test == "train":
            # by default, run 200 epochs
            self.num_epochs = 200
            # use random init model for train
            self.model = torchvision.models.resnet50().to(self.device)
            self.model.train()
            self.criterion = torch.nn.CrossEntropyLoss()
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr,
                             momentum=0.9, weight_decay=5e-4)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        else:
            # use pretrained model for eval
            self.model = torchvision.models.resnet50(pretrained=True).to(self.device)
            self.model.eval()

    def test(self):
        self.model.eval()
        with torch.no_grad():
            pass

    def train(self):
        self.model.train()
        pass

    def eval(self):
        pass