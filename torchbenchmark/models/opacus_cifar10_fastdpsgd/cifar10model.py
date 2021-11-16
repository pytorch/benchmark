from torch import nn

class CIFAR10Model(nn.Module):
    def __init__(self, **_):
        super().__init__()
        self.layer_list = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 32, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.Sequential(nn.Conv2d(32, 32, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.AvgPool2d(2, stride=2),
            nn.Sequential(nn.Conv2d(32, 64, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.Sequential(nn.Conv2d(64, 64, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.AvgPool2d(2, stride=2),
            nn.Sequential(nn.Conv2d(64, 128, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.Sequential(nn.Conv2d(128, 128, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.AvgPool2d(2, stride=2),
            nn.Sequential(nn.Conv2d(128, 256, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.Conv2d(256, 10, (3, 3), padding=1, stride=(1, 1)),
        ])
    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return torch.mean(x, dim=(2, 3))
