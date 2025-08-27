from utils import s3_utils
from utils.python_utils import pip_install_requirements

if __name__ == "__main__":
    pip_install_requirements()
    for filename in ["batch-np-1x.pt", "batch-np-2x.pt"]:
        s3_utils.checkout_s3_data(
            "MODEL_PKLS", f"maml_omniglot/{filename}", decompress=False
        )
