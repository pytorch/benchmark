import os
from torchbenchmark.util.framework.detectron2 import install_detectron2

MODEL_NAME = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.abspath(os.path.dirname(__file__))

if __name__ == '__main__':
    install_detectron2(MODEL_NAME, MODEL_DIR)
