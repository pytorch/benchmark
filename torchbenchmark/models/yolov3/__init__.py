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
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from shlex import split
from .yolo_train import prepare_training_loop
from . import yolo_train

from .yolo_models import *  # set ONNX_EXPORT in models.py
from .yolo_utils.datasets import *
from .yolo_utils.utils import *
from pathlib import Path
from torchbenchmark.tasks import COMPUTER_VISION

class Model:
    task = COMPUTER_VISION.SEGMENTATION
    def __init__(self, device='cpu', jit=False):
        self.device = device
        self.jit = jit

    def get_module(self):
        if self.jit:
            raise NotImplementedError()
        parser = argparse.ArgumentParser()
        root = str(Path(yolo_train.__file__).parent.absolute())
        parser.add_argument('--cfg', type=str, default=f'{root}/cfg/yolov3-spp.cfg', help='*.cfg path')
        parser.add_argument('--names', type=str, default=f'{root}/data/coco.names', help='*.names path')
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
        input = (torch.rand(1, 3, 384, 512).to(opt.device),)
        return model, input
        
    def train(self, niterations=1):
        # the training process is not patched to use scripted models
        if self.jit:
            raise NotImplementedError()

        if self.device == 'cpu':
            raise NotImplementedError("Disabled due to excessively slow runtime - see GH Issue #100")

        root = str(Path(yolo_train.__file__).parent.absolute())
        train_args = split(f"--data {root}/data/coco128.data --img 416 --batch 8 --nosave --notest --epochs 1 --device {self.device} --weights ''")
        print(train_args)
        training_loop = prepare_training_loop(train_args)

        return training_loop(niterations)

    
    def eval(self, niterations=1):
        model, example_inputs = self.get_module()
        img = example_inputs[0]
        im0s_shape = (480, 640, 3)
        for i in range(niterations):
            pred = model(img, augment=False)[0]
            # Apply NMS
            pred = non_max_suppression(pred, 0.3, 0.6,
                                    multi_label=False, classes=None, agnostic=False)

if __name__ == '__main__':
    m = Model(device='cpu', jit=False)
    model, example_inputs = m.get_module()
    model(*example_inputs)
    m.train()
    m.eval()