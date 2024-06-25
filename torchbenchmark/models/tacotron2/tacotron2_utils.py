import os
from pathlib import Path

import numpy as np
import torch
from scipy.io.wavfile import read


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    root = str(Path(__file__).parent)
    # The script is probably run at Meta internally. Attempt to download the dataset from Manifold.
    if not os.path.exists(filename):
        try:
            from torchbenchmark.util.framework.fb.installer import install_data

            root = install_data("tacotron2-minimal")
            filename = os.path.join(
                root,
                "tacotron2-minimal/filelists/ljs_audio_text_train_filelist.txt",
            )
        except Exception as e:
            msg = f"Failed to download data from manifold: {e}"
            raise RuntimeError(msg) from e

    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = []
        for line in f:
            filename, *text = line.strip().split(split)
            filename = f"{root}/{filename}"
            filepaths_and_text.append((filename, *text))
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
