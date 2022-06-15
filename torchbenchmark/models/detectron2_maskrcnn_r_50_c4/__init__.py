import os
from torchbenchmark.tasks import COMPUTER_VISION
from torchbenchmark.util.framework.detectron2.model_factory import Detectron2Model

MODEL_NAME = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.abspath(os.path.dirname(__file__))

class Model(Detectron2Model):
    task = COMPUTER_VISION.SEGMENTATION
    model_file = os.path.join(MODEL_DIR, ".data", f"{MODEL_NAME}.pkl")

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(variant="COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml", test=test, device=device,
                         jit=jit, batch_size=batch_size, extra_args=extra_args)
