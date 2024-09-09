"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import os
import os.path

import torch.utils.data as data

from PIL import Image

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tif",
    ".TIF",
    ".tiff",
    ".TIFF",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    # The script is probably run at Meta internally. Attempt to
    # download the dataset from Manifold.
    if not os.path.isdir(dir):
        try:
            from torchbenchmark.util.framework.fb.installer import install_data

            data_dir = install_data("pytorch_CycleGAN_and_pix2pix_inputs")
            # Process the path to use the data just downloaded.
            dir = os.path.join(data_dir, dir.split("/.data/", 1)[1])
        except Exception as e:
            msg = f"Failed to download data from manifold: {e}"
            raise RuntimeError(msg) from e

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[: min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert("RGB")


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False, loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (
                RuntimeError(
                    "Found 0 images in: " + root + "\n"
                    "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)
                )
            )

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
