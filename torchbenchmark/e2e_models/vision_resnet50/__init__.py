# upstream repo: https://github.com/kuangliu/pytorch-cifar
import torch
import torchvision
import torchvision.transforms as transforms
from torchbenchmark.util.e2emodel import E2EBenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION
import os
from tqdm import tqdm

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
        self.device_num = 1
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
        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.num_examples = len(self.trainloader)

        testset = torchvision.datasets.CIFAR10(
            root=str(data_root), train=False, download=True, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')
        self.lr = 0.1
        # initialize accuracy
        self.accuracy = 0.0

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

    def _test_loop(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for _batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        self.accuracy = 100. * correct / total

    def _train_loop(self):
        for _batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

    def train(self):
        self.model.train()
        # Train num_epochs
        for _epoch in tqdm(range(self.num_epochs), desc = "Training epoch"):
            self._train_loop()
        # calculate total accuracy
        self._test_loop()

    def eval(self):
        raise NotImplementedError("Eval is not yet implemented for this model.")
