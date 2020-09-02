import numpy as np
import random
import torch

# from apex import amp
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config

torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Model:
    def __init__(self, device=None, jit=False):
        self.device = cfg.MODEL.DEVICE = device
        self.jit = jit
        cfg.merge_from_file('configs/e2e_mask_rcnn_R_50_FPN_1x.yaml')
        cfg.merge_from_list(['SOLVER.IMS_PER_BATCH', '2', 
                             'SOLVER.BASE_LR', '0.0025',
                             'SOLVER.MAX_ITER', '720000', 
                             'SOLVER.STEPS', '(480000, 640000)', 
                             'TEST.IMS_PER_BATCH', '1', 
                             'MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN', '2000'])
        cfg.freeze()
        save_config(cfg, "hubconf.config.yml")
        self.module = build_detection_model(cfg)
        start_iter = 0
        is_distributed = False
 
        if self.jit:
            self.module = torch.jit.script(self.module)

        self.module.to(device)

        self.optimizer = make_optimizer(cfg, self.module)
        self.scheduler = make_lr_scheduler(cfg, self.optimizer)
        # Initialize mixed-precision training
        # use_mixed_precision = cfg.DTYPE == "float16"
        # amp_opt_level = 'O1' if use_mixed_precision else 'O0'
        # self.module, self.optimizer = amp.initialize(
            # self.module, self.optimizer,
            # opt_level=amp_opt_level)

        self.data_loader = data_loader = make_data_loader(
            cfg,
            is_train=True,
            is_distributed=is_distributed,
            start_iter=start_iter,
        )

        images, targets, _ = next(iter(data_loader))
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        self.example_inputs = (images, targets)

    def get_module(self):
        return self.module, self.example_inputs

    def eval(self, niter=1):
        self.module.eval()
        for iteration, (images, targets, _) in enumerate(self.data_loader):
            images = images.to(self.device)
            targets = [target.to(self.device) for target in targets]
            self.module(images, targets)
        # for _ in range(niter):
            # self.module(*self.example_inputs)

    def train(self, niter=1):
        self.module.train()
        for _ in range(niter):
            loss_dict = self.module(*self.example_inputs)
            losses = sum(loss for loss in loss_dict.values())
            self.optimizer.zero_grad()

            # Note: If mixed precision is not used, this ends up doing nothing
            # Otherwise apply loss scaling for mixed-precision recipe
            # with amp.scale_loss(losses, self.optimizer) as scaled_losses:
                # scaled_losses.backward()
            losses.backward()

            self.optimizer.step()
            self.scheduler.step()


if __name__ == '__main__':
    m = Model(device='cuda', jit=False)
    module, example_inputs = m.get_module()
    # module(*example_inputs)
    # m.train(niter=1)
    m.eval(niter=1)
