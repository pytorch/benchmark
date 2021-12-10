#!/usr/bin/env python

# Make all randomness deterministic
import random
import argparse
import torch
import os
import numpy as np

random.seed(1337)
torch.manual_seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

from shlex import split
from .yolo_train import prepare_training_loop
from . import yolo_train

from .yolo_models import *  # set ONNX_EXPORT in models.py
from .yolo_utils.datasets import *
from .yolo_utils.utils import *
from pathlib import Path
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION

CURRENT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(CURRENT_DIR.parent.parent, "data", ".data", "coco128")
assert os.path.exists(DATA_DIR), "Couldn't find coco128 data dir, please run install.py again."
class Model(BenchmarkModel):
    task = COMPUTER_VISION.SEGMENTATION
    # Original train batch size: 16
    # Source: https://github.com/ultralytics/yolov3/blob/master/train.py#L447
    def __init__(self, device=None, jit=False, train_bs=16, eval_bs=16):
        super().__init__()
        self.device = device
        self.jit = jit
        # run just 1 epoch
        self.num_epochs = 1
        self.train_num_batch = 1
        self.prefetch = True
        self.train_bs = train_bs
        self.eval_bs = eval_bs
        train_args = split(f"--data {DATA_DIR}/coco128.data --img 416 --batch {train_bs} --nosave --notest \
                             --epochs {self.num_epochs} --device {self.device_str} --weights '' \
                             --train-num-batch {self.train_num_batch} \
                             --prefetch")
        self.training_loop = prepare_training_loop(train_args)
        self.eval_model, self.eval_example_input = self.prep_eval()

    def prep_eval(self):
        parser = argparse.ArgumentParser()
        root = str(Path(yolo_train.__file__).parent.absolute())
        parser.add_argument('--cfg', type=str, default=f'{root}/cfg/yolov3-spp.cfg', help='*.cfg path')
        parser.add_argument('--names', type=str, default=f'{DATA_DIR}/coco.names', help='*.names path')
        parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
        parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
        parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
        parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
        parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
        parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
        parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        opt = parser.parse_args(['--device', self.device])
        opt.cfg = check_file(opt.cfg)  # check file
        opt.names = check_file(opt.names)  # check file
        model = Darknet(opt.cfg, opt.img_size)
        model.to(opt.device).eval()
        example_input = (torch.rand(self.eval_bs, 3, 384, 512).to(self.device),)
        return model, example_input

    def get_module(self):
        if self.jit:
            raise NotImplementedError()
        return self.eval_model, self.eval_example_input

    def train(self, niter=1):
        # the training process is not patched to use scripted models
        if self.jit:
            raise NotImplementedError()
        if self.device == 'cpu':
            raise NotImplementedError("Disabled due to excessively slow runtime - see GH Issue #100")
        return self.training_loop(niter)

    def eval(self, niter=1):
        model, example_inputs = self.get_module()
        for i in range(niter):
            pred = model(*example_inputs, augment=False)[0]
            # Apply NMS
            pred = non_max_suppression(pred, 0.3, 0.6,
                                    multi_label=False, classes=None, agnostic=False)

    @property
    def device_str(self):
        """YoloV3 uses individual GPU indices."""
        return str(
            torch.cuda.current_device() if self.device == "cuda"
            else self.device
        )
