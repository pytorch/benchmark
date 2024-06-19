import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib import request

from utils import s3_utils
from utils.python_utils import pip_install_requirements

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
    "detectron2_fcos_r_50_fpn": None,
}


def install_model_weights(model_name, model_dir):
    assert (
        model_name in MODEL_WEIGHTS_MAP
    ), f"Model {model_name} is not in MODEL_WEIGHTS_MAP. Cannot download the model weights file."
    model_full_path = Path(os.path.join(model_dir, ".data", f"{model_name}.pkl"))
    if model_name in MODEL_WEIGHTS_MAP and MODEL_WEIGHTS_MAP[model_name]:
        # download the file if not exists
        # TODO: verify the model file integrity
        if os.path.exists(model_full_path):
            return
        model_full_path.parent.mkdir(parents=True, exist_ok=True)
        request.urlretrieve(MODEL_WEIGHTS_MAP[model_name], model_full_path)


def pip_install_requirements_detectron2():
    requirements_file = os.path.join(CURRENT_DIR, "requirements.txt")
    # Installing by --no-build-isolation after explicitly installing build-time requirements is required.
    # See https://github.com/facebookresearch/detectron2/issues/4921
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "wheel", "cython"] # Build-time requirements
    )
    pip_install_requirements(requirements_txt=requirements_file, no_build_isolation=True)


# This is to workaround https://github.com/facebookresearch/detectron2/issues/3934
def remove_tools_directory():
    try:
        import detectron2
        import tools

        d2_dir_path = Path(detectron2.__file__).parent
        assumed_tools_path = d2_dir_path.parent.joinpath("tools")
        if tools.__file__ and assumed_tools_path.exists():
            shutil.rmtree(str(assumed_tools_path))
    except ImportError:
        # if the "tools" package doesn't exist, do nothing
        pass


def install_detectron2(model_name, model_dir):
    s3_utils.checkout_s3_data(
        "INPUT_TARBALLS", "coco2017-minimal.tar.gz", decompress=True
    )
    install_model_weights(model_name, model_dir)
    pip_install_requirements_detectron2()
    remove_tools_directory()
