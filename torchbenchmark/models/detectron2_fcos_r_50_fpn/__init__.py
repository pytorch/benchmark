import os
from torchbenchmark.tasks import COMPUTER_VISION
from torchbenchmark.util.framework.detectron2.model_factory import Detectron2Model

MODEL_NAME = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.abspath(os.path.dirname(__file__))

class Model(Detectron2Model):
    task = COMPUTER_VISION.SEGMENTATION
    model_file = None
    # A hack to workaround fcos model instantiate error
    FCOS_USE_BN = True

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(variant="COCO-Detection/fcos_R_50_FPN_1x.py", test=test, device=device,
                         jit=jit, batch_size=batch_size, extra_args=extra_args)
