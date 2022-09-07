import argparse
import random
import torch

import numpy as np

from torchbenchmark.util.env_check import set_random_seed

from dalle2_pytorch import DALLE2, Unet, Decoder, DiffusionPriorNetwork, DiffusionPrior, OpenAIClipAdapter
import typing

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
from pathlib import Path
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION

import io


class Model(BenchmarkModel):
    task = COMPUTER_VISION.GENERATION
    DEFAULT_TRAIN_BSIZE = 4
    DEFAULT_EVAL_BSIZE = 4

    def __init__(self, test, device, batch_size=None, jit=False, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

        if device == "cpu":
            raise NotImplementedError("DALL-E 2 Not Supported on CPU")
    
        self.clip = OpenAIClipAdapter().cuda()

        self.sample_text = torch.randint(0, 49408, (4, 256)).cuda()
        self.sample_images = torch.randn(4, 3, 256, 256).cuda()

        prior_network = DiffusionPriorNetwork(
            dim = 512,
            depth = 6,
            dim_head = 64,
            heads = 8
        ).cuda()

        self.diffusion_prior = DiffusionPrior(
            net = prior_network,
            clip = self.clip,
            timesteps = 100,
            cond_drop_prob = 0.2
        ).cuda()

        unet1 = Unet(
            dim = 128,
            image_embed_dim = 512,
            cond_dim = 128,
            channels = 3,
            dim_mults=(1, 2, 4, 8),
            text_embed_dim = 512,
            cond_on_text_encodings = True  # set to True for any unets that need to be conditioned on text encodings (ex. first unet in cascade)
        ).cuda()

        unet2 = Unet(
            dim = 16,
            image_embed_dim = 512,
            cond_dim = 128,
            channels = 3,
            dim_mults = (1, 2, 4, 8, 16)
        ).cuda()

        self.decoder = Decoder(
            unet = (unet1, unet2),
            image_sizes = (128, 256),
            clip = self.clip,
            timesteps = 1000,
            sample_timesteps = (250, 27),
            image_cond_drop_prob = 0.1,
            text_cond_drop_prob = 0.5
        ).cuda()

        self.model = DALLE2(prior = self.diffusion_prior, decoder = self.decoder).cuda()
        self.example_inputs = self.sample_text = torch.randint(0, 49408, (4, 256)).cuda()

        if test == "train":
            self.diffusion_prior.train()
            self.decoder.train()
        elif test == "eval":
            self.diffusion_prior.eval()
            self.decoder.eval()

    def get_module(self):
        return self.model, self.example_inputs

    def set_module(self, new_model):
        self.diffusion_prior, self.decoder = new_model

    def eval(self):
        inputs, model = self.example_inputs, self.model
        images = model(inputs)
        return images

    def train(self):
        # openai pretrained clip - defaults to ViT-B/32
        clip = self.clip

        # prior networks (with transformer)
        diffusion_prior = self.diffusion_prior

        loss = diffusion_prior(self.sample_text, self.sample_images)
        loss.backward()

        # decoder (with unet)
        decoder = self.decoder

        loss = decoder(self.sample_images, self.sample_text, unet_number=1)
        loss.backward()

        loss = decoder(self.sample_images, self.sample_text, unet_number=2)
        loss.backward()