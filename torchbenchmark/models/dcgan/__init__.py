# Ported from pytorch example:
#   https://github.com/pytorch/examples/blob/master/dcgan/main.py


from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

from pathlib import Path

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION

class DCGAN:
 def __init__(self, bench):

  # Spatial size of training images. All images will be resized to this
  #   size using a transformer.
  self.image_size = 64

  # Number of channels in the training images. For color images this is 3
  self.nc = 3

  # Size of z latent vector (i.e. size of generator input)
  self.nz = 100

  # Size of feature maps in generator
  self.ngf = 64

  # Size of feature maps in discriminator
  self.ndf = 64

  # Number of training epochs
  self.num_epochs = 5

  # Learning rate for optimizers
  self.lr = 0.0002

  # Beta1 hyperparam for Adam optimizers
  self.beta1 = 0.5

  # Number of GPUs available. Use 0 for CPU mode.
  self.ngpu = 1

  self.device = bench.device

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, dcgan):
        super(Generator, self).__init__()
        self.ngpu = dcgan.ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( dcgan.nz, dcgan.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(dcgan.ngf * 8),
            nn.ReLU(True),
            # state size. (dcgan.ngf*8) x 4 x 4
            nn.ConvTranspose2d(dcgan.ngf * 8, dcgan.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dcgan.ngf * 4),
            nn.ReLU(True),
            # state size. (dcgan.ngf*4) x 8 x 8
            nn.ConvTranspose2d( dcgan.ngf * 4, dcgan.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dcgan.ngf * 2),
            nn.ReLU(True),
            # state size. (dcgan.ngf*2) x 16 x 16
            nn.ConvTranspose2d( dcgan.ngf * 2, dcgan.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dcgan.ngf),
            nn.ReLU(True),
            # state size. (dcgan.ngf) x 32 x 32
            nn.ConvTranspose2d( dcgan.ngf, dcgan.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (dcgan.nc) x 64 x 64
        )

        self.jt = None
        self.jitshape = None
        self.debug_print = False

    def forward(self, input):

      if self.debug_print:
        print(input.shape)

      return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ncgan):
        ngpu = ncgan.ngpu
        nc = ncgan.nc
        ndf = ncgan.ndf

        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.jt = None
        self.jitshape = None

    def forward(self, input):
      return self.main(input)

class Model(BenchmarkModel):
    task = COMPUTER_VISION.GENERATION
    DEFAULT_TRAIN_BSIZE = 32
    DEFAULT_EVAL_BSIZE = 256

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        self.debug_print = False

        self.root = str(Path(__file__).parent)
        self.dcgan = DCGAN(self)

        dcgan = self.dcgan

        device = dcgan.device
        ngpu = dcgan.ngpu
        nz = dcgan.nz
        lr = dcgan.lr
        beta1 = dcgan.beta1
        num_epochs = dcgan.num_epochs

        # Create the generator
        self.netG = Generator(dcgan).to(device)

        # Handle multi-gpu if desired
        if (dcgan.device == 'cuda') and (ngpu > 1):
            self.netG = nn.DataParallel(self.netG, list(range(ngpu)))

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        self.netG.apply(weights_init)

        if self.debug_print:
            # Print the model
            print(self.netG)

        # Create the Discriminator
        netD = Discriminator(dcgan).to(device)

        # Handle multi-gpu if desired
        if (dcgan.device == 'cuda') and (ngpu > 1):
            netD = nn.DataParallel(self.netD, list(range(ngpu)))

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        netD.apply(weights_init)
        
        if self.debug_print:
            # Print the model
            print(netD)

        # Initialize BCELoss function
        self.criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        self.fixed_noise = torch.randn(64, nz, 1, 1, device=device)

        # Establish convention for real and fake labels during training
        self.real_label = 1.
        self.fake_label = 0.

        # Random values as surrogate for batch of photos
        self.example_inputs = torch.randn(self.batch_size, 3, 64, 64, device=self.device)
        self.model = netD
        if test == "train":
            # Setup Adam optimizers for both G and D
            self.optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
            self.optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
        elif test == "eval":
            # inference would just run descriminator so thats what we'll do too.
            self.inference_just_descriminator = True
            if False == self.inference_just_descriminator:
                self.eval_noise = torch.randn(self.batch_size, nz, 1, 1, device=self.device)

    def jit_callback(self):
        assert self.jit, "Calling JIT callback without specifying the JIT option."
        self.model = torch.jit.trace(self.model,(self.example_inputs,))
        if self.test == "eval" and False == self.inference_just_descriminator:
            self.netG = torch.jit.trace(self.netG,(self.eval_noise,))

    def get_module(self):
        return self.model, (self.example_inputs,)

    def eval(self):
       if False == self.inference_just_descriminator:
           # Generate fake image batch with G
           self.eval_fake = self.netG(self.eval_noise)

       # Since we just updated D, perform another forward pass of all-fake batch through D
       output = self.model(self.example_inputs).view(-1)
       return (output, )

    def train(self):

        # Training Loop

        # Lists to keep track of progress
        img_list = []
        iters = 0

        dcgan = self.dcgan
        device = dcgan.device

        num_epochs = dcgan.num_epochs
        num_train_batch = 1

        lr = dcgan.lr
        nz = dcgan.nz
        beta1 = dcgan.beta1

        netD = self.model
        netG = self.netG

        criterion = self.criterion
        optimizerD = self.optimizerD
        optimizerG = self.optimizerG

        real_label = self.real_label
        fake_label = self.fake_label

        benchmark_pic = self.example_inputs

        # For each epoch
        for epoch in range(num_epochs):

            for i in range(num_train_batch):

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                netD.zero_grad()
                # Format batch

                real_cpu = benchmark_pic
                b_size = real_cpu.size(0)

                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                # Forward pass real batch through D
                output = netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                # Generate fake image batch with G
                fake = netG(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()
