from torchbenchmark.tasks import COMPUTER_VISION
from torchbenchmark.util.framework.detectron2 import Detectron2Model

class Model(Detectron2Model):
    task = COMPUTER_VISION.SEGMENTATION
    DEFAULT_TRAIN_BSIZE = 1
    DEFAULT_EVAL_BSIZE = 2

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(variant="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml", test=test, device=device,
                         jit=jit, batch_size=batch_size, extra_args=extra_args)
