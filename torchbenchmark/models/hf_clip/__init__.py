# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...util.model import BenchmarkModel
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel


class ContrastiveLossWithTemperature(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLossWithTemperature, self).__init__()
        self.temperature = temperature

    def forward(self, image_embeddings, text_embeddings):
        # Ensure batch sizes are equal
        assert image_embeddings.size(0) == text_embeddings.size(
            0
        ), "Batch sizes of image and text embeddings should be the same"

        # Compute the similarity between image and text embeddings
        logits = torch.matmul(image_embeddings, text_embeddings.T) / self.temperature

        # Compute the labels for the positive pairs
        labels = torch.arange(logits.size(0)).to(image_embeddings.device)

        # Compute the contrastive loss
        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        return loss / 2


class Model(BenchmarkModel):
    DEFAULT_EVAL_BSIZE = 32
    DEFAULT_TRAIN_BSIZE = 32

    def __init__(self, test, device, batch_size=1, extra_args=[]):
        super().__init__(
            test=test, device=device, batch_size=batch_size, extra_args=extra_args
        )
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            self.device
        )
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        text = "the dog is here"
        images = [image] * self.batch_size
        texts = [text] * self.batch_size
        self.inputs = processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )

        # dict_keys(['input_ids', 'attention_mask', 'pixel_values'])
        for key in self.inputs:
            self.inputs[key] = self.inputs[key].to(self.device)

        # Add the loss function and optimizer
        self.loss_fn = ContrastiveLossWithTemperature()
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.loss_fn.parameters()),
            lr=5.0e-4,
            weight_decay=1.0e-4,
            eps=1.0e-6,
        )

    def get_module(self):
        return self.model, self.inputs

    def train(self):
        image_tensor = self.inputs["pixel_values"]
        text_tensor = self.inputs["input_ids"]
        total_loss = 0
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(**self.inputs)
        image_embedding = outputs.image_embeds
        text_embedding = outputs.text_embeds

        # Compute the loss
        loss = self.loss_fn(image_embedding, text_embedding)
        loss.backward()
        self.optimizer.step()

        total_loss += loss.item()

        # Return the average loss
        return total_loss / len(text_tensor)

    def eval(self):
        return self.model(**self.inputs)
