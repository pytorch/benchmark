# This file was adapted from
# https://github.com/facebookresearch/higher/blob/master/examples/maml-omniglot.py
# It comes with the following license.
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple
import higher

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import OTHER


class Model(BenchmarkModel):
    task = OTHER.OTHER_TASKS
    # batch size in the traditional sense doesn't apply to this maml model.
    # Instead, there is a task number (32 in this case) and K-shot
    # (5 in this case) and these were chosen to be representative of
    # the training done in the original MAML paper
    # (https://arxiv.org/pdf/1703.03400.pdf)
    #
    # The goal of MAML is to train a model in a way that if one brings along
    # a new task and K data points, then the model generalizes well on the
    # test set for that task.
    #
    # The task number (also known as the meta-batch size) is the number of
    # independent tasks the model gets trained on.
    # K-shot means that each task only sees K data points.
    #
    # We've set the following variables to be equal to the task number.
    DEFAULT_TRAIN_BSIZE = 5
    DEFAULT_EVAL_BSIZE = 5
    ALLOW_CUSTOMIZE_BSIZE = False

    # TODO: There _should_ be a way to plug in an optim here, but this
    # can be a next step. For now, the optim is not customizable.
    CANNOT_SET_CUSTOM_OPTIMIZER = True

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, batch_size=batch_size, extra_args=extra_args)

        n_way = 5
        net = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64, n_way)).to(device)
        self.model = net

        root = str(Path(__file__).parent)
        with torch.serialization.safe_globals(
            [
                np.core.multiarray._reconstruct,
                np.ndarray,
                np.dtype,
                (
                    type(np.dtype(np.float32))
                    if np.__version__ < "1.25.0"
                    else np.dtypes.Float32DType
                ),
                (
                    type(np.dtype(np.int64))
                    if np.__version__ < "1.25.0"
                    else np.dtypes.Int64DType
                ),
            ]
        ):
            self.meta_inputs = torch.load(f"{root}/batch.pt", weights_only=True)
        self.meta_inputs = tuple(
            [torch.from_numpy(i).to(self.device) for i in self.meta_inputs]
        )
        self.example_inputs = (self.meta_inputs[0][0],)

    def get_module(self):
        return self.model, self.example_inputs

    def train(self):
        net, _ = self.get_module()
        net.train()
        x_spt, y_spt, x_qry, y_qry = self.meta_inputs
        meta_opt = optim.Adam(net.parameters(), lr=1e-3)

        if True:
            task_num, setsz, c_, h, w = x_spt.size()
            querysz = x_qry.size(1)

            n_inner_iter = 5
            inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)

            meta_opt.zero_grad()
            for i in range(task_num):
                with higher.innerloop_ctx(
                    net, inner_opt, copy_initial_weights=False
                ) as (fnet, diffopt):
                    for _ in range(n_inner_iter):
                        spt_logits = fnet(x_spt[i])
                        spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                        diffopt.step(spt_loss)

                    qry_logits = fnet(x_qry[i])
                    qry_loss = F.cross_entropy(qry_logits, y_qry[i])
                    qry_loss.backward()

            meta_opt.step()

    def eval(self) -> Tuple[torch.Tensor]:
        model, (example_input,) = self.get_module()
        model.eval()
        with torch.no_grad():
            out = model(example_input)
        return (out, )
