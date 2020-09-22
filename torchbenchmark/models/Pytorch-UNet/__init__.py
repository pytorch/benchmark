import torch
from .unet import UNet as _UNet
from torch import optim
import torch.nn as nn
from .utils.dataset import BasicDataset
from torch.utils.data import DataLoader
from pathlib import Path

torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Model:
    def __init__(self, device=None, batch_size=1, img_scale=0.5, jit=False):
        self.device = device
        self.jit = jit

        root = Path(__file__).parent

        self.model = _UNet(n_channels=3, n_classes=1, bilinear=True)
        checkpoint = 'https://github.com/milesial/Pytorch-UNet/releases/download/v1.0/unet_carvana_scale1_epoch5.pth'
        self.model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))

        dir_img = str(root / 'data' / 'imgs')
        dir_mask = str(root / 'data' / 'masks' )

        dataset = BasicDataset(dir_img, dir_mask, img_scale)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        batch = next(iter(train_loader))
        self.imgs = batch['image']
        self.true_masks = batch['mask']

        if self.jit:
            self.model = torch.jit.script(self.model)

        if self.model.n_classes > 1:
            self.criterion = nn.CrossEntropyLoss().to(device)
        else:
            self.criterion = nn.BCEWithLogitsLoss().to(device)

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.001, weight_decay=1e-8, momentum=0.9)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min' if self.model.n_classes > 1 else 'max', patience=2)


    def get_module(self):
        return self.model, (self.imgs, )

    def eval(self, niter=1):
        # TODO: consider evaling a full image
        with torch.no_grad():
            for _ in range(niter):
                output = self.model(self.text, self.offsets)
                loss = self.criterion(output, self.cls)

    def train(self, niter=1):
        for _ in range(niter):
            self.optimizer.zero_grad()
            masks_pred = self.model(self.imgs)
            loss = self.criterion(masks_pred, self.true_masks)
            loss.backward()
            nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = Model(device=device, jit=False)
    model, example_inputs = m.get_module()
    model(*example_inputs)
    m.train()
    m.eval()