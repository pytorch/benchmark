from torchbenchmark.util.framework.timm.model_factory import TimmModel
from torchbenchmark.tasks import COMPUTER_VISION

class Model(TimmModel):
    task = COMPUTER_VISION.CLASSIFICATION

    # Original train batch size 128, hardware Nvidia rtx 3090
    # Source: https://gist.github.com/rwightman/bb59f9e245162cee0e38bd66bd8cd77f#file-bench_by_train-csv-L147
    # Eval batch size 256, hardware Nvidia rtx 3090
    # Source: https://github.com/rwightman/pytorch-image-models/blob/f7d210d759beb00a3d0834a3ce2d93f6e17f3d38/results/model_benchmark_amp_nchw_rtx3090.csv
    # Downscale to 128 to fit T4
    DEFAULT_TRAIN_BSIZE = 128
    DEFAULT_EVAL_BSIZE = 128

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, model_name='dm_nfnet_f0', device=device,
                         jit=jit, batch_size=batch_size, extra_args=extra_args)
