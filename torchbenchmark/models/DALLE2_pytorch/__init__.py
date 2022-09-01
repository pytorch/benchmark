import argparse
import random
import torch

import numpy as np

from torchbenchmark.util.env_check import set_random_seed

from .dalle2_pytorch import DALLE2, Unet, Decoder, DiffusionPriorNetwork, DiffusionPrior, OpenAIClipAdapter
import typing

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
from pathlib import Path
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION

import io


class Model(BenchmarkModel):
    task = COMPUTER_VISION.GENERATION
    DEFAULT_TRAIN_BSIZE = 16
    DEFAULT_EVAL_BSIZE = 16

    def __init__(self, test, device, batch_size=None, jit=False, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
    
        self.clip = OpenAIClipAdapter()

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

        if test == "train":
            self.train()
        elif test == "eval":
            self.eval()


    def get_module(self):
        return (self.diffusion_prior, self.decoder,), ((self.sample_text, self.sample_images), (self.sample_images, self.sample_text),)

    def set_module(self, new_model):
        self.diffusion_prior, self.decoder = new_model

    def eval(self):
        diffusion_prior = self.diffusion_prior
        decoder = self.decoder

        dalle2 = DALLE2(
            prior = diffusion_prior,
            decoder = decoder
        )

        texts = ['glistening morning dew on a flower petal']
        images = dalle2(texts)

        return images

    def train(self):
        # openai pretrained clip - defaults to ViT-B/32
        clip = self.clip

        # mock data
        text = torch.randint(0, 49408, (4, 256)).cuda()
        images = torch.randn(4, 3, 256, 256).cuda()

        # prior networks (with transformer)
        diffusion_prior = self.diffusion_prior

        loss = diffusion_prior(text, images)
        loss.backward()

        # decoder (with unet)
        decoder = self.decoder

        loss = decoder(images, text, unet_number=1)
        loss.backward()

        loss = decoder(images, text, unet_number=2)
        loss.backward()