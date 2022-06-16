import os
import sys
import subprocess
from pathlib import Path
from urllib import request

CURRENT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
# Load pre-trained weights
# copied from https://github.com/facebookresearch/detectron2/blob/5934a1452801e669bbf9479ae222ce1a8a51f52e/MODEL_ZOO.md
MODEL_WEIGHTS_MAP = {
    "detectron2_fasterrcnn_r_50_c4": "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/model_final_721ade.pkl",
    "detectron2_fasterrcnn_r_50_dc5": "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_DC5_1x/137847829/model_final_51d356.pkl",
    "detectron2_fasterrcnn_r_50_fpn": "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_1x/137257794/model_final_b275ba.pkl",
    "detectron2_fasterrcnn_r_101_c4": "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_C4_3x/138204752/model_final_298dad.pkl",
    "detectron2_fasterrcnn_r_101_dc5": "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_DC5_3x/138204841/model_final_3e0943.pkl",
    "detectron2_fasterrcnn_r_101_fpn": "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl",
    "detectron2_maskrcnn_r_50_c4": "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x/137259246/model_final_9243eb.pkl",
    "detectron2_maskrcnn_r_50_fpn": "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl",
    "detectron2_maskrcnn_r_101_c4": "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x/138363239/model_final_a2914c.pkl",
    "detectron2_maskrcnn_r_101_fpn": "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl",
    "detectron2_maskrcnn": "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl",
    # this model uses randomly initialized weights, so no weights file is required
    "detectron2_fcos_r_50_fpn": "",
}

def check_data_dir():
    coco2017_data_dir = os.path.join(CURRENT_DIR.parent.parent.parent, "data", ".data", "coco2017-minimal")
    assert os.path.exists(coco2017_data_dir), "Couldn't find coco2017 minimal data dir, please run install.py again."

def install_model_weights(model_name, model_dir):
    assert model_name in MODEL_WEIGHTS_MAP, f"Model {model_name} is not in MODEL_WEIGHTS_MAP. Cannot download the model weights file."
    model_full_path = Path(os.path.join(model_dir, ".data", f"{model_name}.pkl"))
    if MODEL_WEIGHTS_MAP[model_name]:
        # download the file if not exists
        # TODO: verify the model file integrity
        if os.path.exists(model_full_path):
            return
        model_full_path.parent.mkdir(parents=True, exist_ok=True)
        request.urlretrieve(MODEL_WEIGHTS_MAP[model_name], model_full_path)

def pip_install_requirements():
    requirements_file = os.path.join(CURRENT_DIR, "requirements.txt")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', requirements_file])

def install_detectron2(model_name, model_dir):
    check_data_dir()
    install_model_weights(model_name, model_dir)
    pip_install_requirements()