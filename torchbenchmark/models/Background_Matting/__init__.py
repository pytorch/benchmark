import os
import time
from argparse import Namespace
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from .data_loader import VideoData
from .functions import compose_image_withshift, write_tb_log
from .networks import ResnetConditionHR, MultiscaleDiscriminator, conv_init
from .loss_functions import alpha_loss, compose_loss, alpha_gradient_loss, GANloss
import random
import numpy as np
from pathlib import Path
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION
from torchbenchmark import DATA_PATH

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

def _collate_filter_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def _create_data_dir():
    data_dir = Path(__file__).parent.joinpath(".data")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

class Model(BenchmarkModel):
    task = COMPUTER_VISION.OTHER_COMPUTER_VISION
    # Original btach size: 4
    # Original hardware: unknown
    # Source: https://arxiv.org/pdf/2004.00626.pdf
    DEFAULT_TRAIN_BSIZE = 4
    DEFAULT_EVAL_BSIZE = 1
    ALLOW_CUSTOMIZE_BSIZE = False

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

        self.opt = Namespace(**{
            'n_blocks1': 7,
            'n_blocks2': 3,
            'batch_size': self.batch_size,
            'resolution': 512,
            'name': 'Real_fixed'
        })

        datadir = os.path.join(DATA_PATH, "Background_Matting_inputs")
        csv_file_path = _create_data_dir().joinpath("Video_data_train_processed.csv")
        with open(f"{datadir}/Video_data_train.csv", "r") as r:
            with open(csv_file_path, "w") as w:
                w.write(r.read().format(scriptdir=datadir))
        data_config_train = {
            'reso': (self.opt.resolution, self.opt.resolution)}
        traindata = VideoData(csv_file=csv_file_path,
                              data_config=data_config_train, transform=None)
        train_loader = torch.utils.data.DataLoader(
            traindata, batch_size=self.opt.batch_size, shuffle=True, num_workers=0, collate_fn=_collate_filter_none)
        self.train_data = []
        for data in train_loader:
            self.train_data.append(data)
            for key in data:
                data[key].to(self.device)

        netB = ResnetConditionHR(input_nc=(
            3, 3, 1, 4), output_nc=4, n_blocks1=self.opt.n_blocks1, n_blocks2=self.opt.n_blocks2)
        netB.to(self.device)
        netB.eval()
        for param in netB.parameters():  # freeze netB
            param.requires_grad = False
        self.netB = netB

        netG = ResnetConditionHR(input_nc=(
            3, 3, 1, 4), output_nc=4, n_blocks1=self.opt.n_blocks1, n_blocks2=self.opt.n_blocks2)
        netG.apply(conv_init)
        self.netG = netG

        self.netG.to(self.device)

        netD = MultiscaleDiscriminator(
            input_nc=3, num_D=1, norm_layer=nn.InstanceNorm2d, ndf=64)
        netD.apply(conv_init)
        # netD = nn.DataParallel(netD)
        self.netD = netD
        self.netD.to(self.device)

        self.l1_loss = alpha_loss()
        self.c_loss = compose_loss()
        self.g_loss = alpha_gradient_loss()
        self.GAN_loss = GANloss()

        self.optimizerG = optim.Adam(netG.parameters(), lr=1e-4)
        self.optimizerD = optim.Adam(netD.parameters(), lr=1e-5)

        self.log_writer = SummaryWriter(datadir)
        self.model_dir = datadir

        self._maybe_trace()

    def _maybe_trace(self):
        for data in self.train_data:
            bg, image, seg, multi_fr = data['bg'], data['image'], data['seg'], data['multi_fr']
            bg, image, seg, multi_fr = Variable(bg.to(self.device)), Variable(
                image.to(self.device)), Variable(seg.to(self.device)), Variable(multi_fr.to(self.device))
            if self.jit:
                self.netB = torch.jit.trace(
                    self.netB, (image, bg, seg, multi_fr))
                self.netG = torch.jit.trace(
                    self.netG, (image, bg, seg, multi_fr))
            else:
                self.netB(image, bg, seg, multi_fr)
                self.netG(image, bg, seg, multi_fr)
            break

    def get_module(self):
        # use netG (generation) for the return module
        for _i, data in enumerate(self.train_data):
            bg, image, seg, multi_fr, seg_gt, back_rnd = data['bg'], data[
                'image'], data['seg'], data['multi_fr'], data['seg-gt'], data['back-rnd']
            return self.netG, (image.to(self.device), bg.to(self.device), seg.to(self.device), multi_fr.to(self.device))

    def train(self):
        self.netG.train()
        self.netD.train()
        lG, lD, GenL, DisL_r, DisL_f, alL, fgL, compL, elapse_run, elapse = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        t0 = time.time()
        KK = len(self.train_data)
        wt = 1
        epoch = 0
        step = 50
        num_of_batches = 1

        for i, data in zip(range(num_of_batches), self.train_data):
            # Initiating

            bg, image, seg, multi_fr, seg_gt, back_rnd = data['bg'], data[
                'image'], data['seg'], data['multi_fr'], data['seg-gt'], data['back-rnd']

            bg, image, seg, multi_fr, seg_gt, back_rnd = Variable(bg.to(self.device)), Variable(image.to(self.device)), Variable(
                seg.to(self.device)), Variable(multi_fr.to(self.device)), Variable(seg_gt.to(self.device)), Variable(back_rnd.to(self.device))
            mask0 = Variable(torch.ones(seg.shape).to(self.device))

            tr0 = time.time()

            # pseudo-supervision
            alpha_pred_sup, fg_pred_sup = self.netB(image, bg, seg, multi_fr)
            if self.device == 'cuda':
                mask = (alpha_pred_sup > -0.98).type(torch.cuda.FloatTensor)
                mask1 = (seg_gt > 0.95).type(torch.cuda.FloatTensor)
            else:
                mask = (alpha_pred_sup > -0.98).type(torch.FloatTensor)
                mask1 = (seg_gt > 0.95).type(torch.FloatTensor)

            # Train Generator

            alpha_pred, fg_pred = self.netG(image, bg, seg, multi_fr)

            # pseudo-supervised losses
            al_loss = self.l1_loss(alpha_pred_sup, alpha_pred, mask0) + \
                0.5 * self.g_loss(alpha_pred_sup, alpha_pred, mask0)
            fg_loss = self.l1_loss(fg_pred_sup, fg_pred, mask)

            # compose into same background
            comp_loss = self.c_loss(image, alpha_pred, fg_pred, bg, mask1)

            # randomly permute the background
            perm = torch.LongTensor(np.random.permutation(bg.shape[0]))
            bg_sh = bg[perm, :, :, :]

            if self.device == 'cuda':
                al_mask = (alpha_pred > 0.95).type(torch.cuda.FloatTensor)
            else:
                al_mask = (alpha_pred > 0.95).type(torch.FloatTensor)

            # Choose the target background for composition
            # back_rnd: contains separate set of background videos captured
            # bg_sh: contains randomly permuted captured background from the same minibatch
            if np.random.random_sample() > 0.5:
                bg_sh = back_rnd

            image_sh = compose_image_withshift(
                alpha_pred, image*al_mask + fg_pred*(1-al_mask), bg_sh, seg)

            fake_response = self.netD(image_sh)

            loss_ganG = self.GAN_loss(fake_response, label_type=True)

            lossG = loss_ganG + wt*(0.05*comp_loss+0.05*al_loss+0.05*fg_loss)

            self.optimizerG.zero_grad()

            lossG.backward()
            self.optimizerG.step()

            # Train Discriminator

            fake_response = self.netD(image_sh)
            real_response = self.netD(image)

            loss_ganD_fake = self.GAN_loss(fake_response, label_type=False)
            loss_ganD_real = self.GAN_loss(real_response, label_type=True)

            lossD = (loss_ganD_real+loss_ganD_fake)*0.5

            # Update discriminator for every 5 generator update
            if i % 5 == 0:
                self.optimizerD.zero_grad()
                lossD.backward()
                self.optimizerD.step()

            lG += lossG.data
            lD += lossD.data
            GenL += loss_ganG.data
            DisL_r += loss_ganD_real.data
            DisL_f += loss_ganD_fake.data

            alL += al_loss.data
            fgL += fg_loss.data
            compL += comp_loss.data

            self.log_writer.add_scalar(
                'Generator Loss', lossG.data, epoch*KK + i + 1)
            self.log_writer.add_scalar('Discriminator Loss',
                                       lossD.data, epoch*KK + i + 1)
            self.log_writer.add_scalar('Generator Loss: Fake',
                                       loss_ganG.data, epoch*KK + i + 1)
            self.log_writer.add_scalar('Discriminator Loss: Real',
                                       loss_ganD_real.data, epoch*KK + i + 1)
            self.log_writer.add_scalar('Discriminator Loss: Fake',
                                       loss_ganD_fake.data, epoch*KK + i + 1)

            self.log_writer.add_scalar('Generator Loss: Alpha',
                                       al_loss.data, epoch*KK + i + 1)
            self.log_writer.add_scalar('Generator Loss: Fg',
                                       fg_loss.data, epoch*KK + i + 1)
            self.log_writer.add_scalar('Generator Loss: Comp',
                                       comp_loss.data, epoch*KK + i + 1)

            t1 = time.time()

            elapse += t1 - t0
            elapse_run += t1-tr0
            t0 = t1

            if i % step == (step-1):
                print('[%d, %5d] Gen-loss:  %.4f Disc-loss: %.4f Alpha-loss: %.4f Fg-loss: %.4f Comp-loss: %.4f Time-all: %.4f Time-fwbw: %.4f' %
                      (epoch + 1, i + 1, lG/step, lD/step, alL/step, fgL/step, compL/step, elapse/step, elapse_run/step))
                lG, lD, GenL, DisL_r, DisL_f, alL, fgL, compL, elapse_run, elapse = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

                write_tb_log(image, 'image', self.log_writer, i)
                write_tb_log(seg, 'seg', self.log_writer, i)
                write_tb_log(alpha_pred_sup, 'alpha-sup', self.log_writer, i)
                write_tb_log(alpha_pred, 'alpha_pred', self.log_writer, i)
                write_tb_log(fg_pred_sup*mask, 'fg-pred-sup',
                             self.log_writer, i)
                write_tb_log(fg_pred*mask, 'fg_pred', self.log_writer, i)

                # composition
                alpha_pred = (alpha_pred+1)/2
                comp = fg_pred*alpha_pred + (1-alpha_pred)*bg
                write_tb_log(comp, 'composite-same', self.log_writer, i)
                write_tb_log(image_sh, 'composite-diff', self.log_writer, i)

                del comp

            del mask, back_rnd, mask0, seg_gt, mask1, bg, alpha_pred, alpha_pred_sup, image, fg_pred_sup, fg_pred, seg, multi_fr, image_sh, bg_sh, fake_response, real_response, al_loss, fg_loss, comp_loss, lossG, lossD, loss_ganD_real, loss_ganD_fake, loss_ganG

        if (epoch % 2 == 0):
            torch.save(self.netG.state_dict(),
                       os.path.join(self.model_dir, 'netG_epoch_%d.pth' % (epoch)))
            torch.save(self.optimizerG.state_dict(),
                       os.path.join(self.model_dir, 'optimG_epoch_%d.pth' % (epoch)))
            torch.save(self.netD.state_dict(),
                       os.path.join(self.model_dir, 'netD_epoch_%d.pth' % (epoch)))
            torch.save(self.optimizerD.state_dict(),
                       os.path.join(self.model_dir, 'optimD_epoch_%d.pth' % (epoch)))

            # Change weight every 2 epoch to put more stress on discriminator weight and less on pseudo-supervision
            wt = wt/2

    def eval(self):
        raise NotImplementedError()
