import os
from torchbenchmark.tasks import COMPUTER_VISION
from torchbenchmark.util.framework.detectron2.model_factory import Detectron2Model

MODEL_NAME = os.path.basename(os.path.dirname(__file__))
MODEL_DIR = os.path.abspath(os.path.dirname(__file__))

class Model(Detectron2Model):
    task = COMPUTER_VISION.DETECTION
    model_file = os.path.join(MODEL_DIR, ".data", f"{MODEL_NAME}.pkl")

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(variant="COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml", test=test, device=device,
                         batch_size=batch_size, extra_args=extra_args)
