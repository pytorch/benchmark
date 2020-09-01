import os
import time
from argparse import Namespace
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from data_loader import VideoData
from functions import compose_image_withshift, write_tb_log
from networks import ResnetConditionHR, MultiscaleDiscriminator, conv_init
from loss_functions import alpha_loss, compose_loss, alpha_gradient_loss, GANloss
import random
import numpy as np

torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _collate_filter_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class Model:
    def __init__(self, device=None, jit=True):
        self.device = device
        self.jit = jit
        self.opt = Namespace(**{
            'n_blocks1': 7,
            'n_blocks2': 3,
            'batch_size': 1,
            'resolution': 512,
            'name': 'Real_fixed'
        })

        scriptdir = os.path.dirname(os.path.realpath(__file__))
        csv_file = "Video_data_train_processed.csv"
        with open("Video_data_train.csv", "r") as r:
            with open(csv_file, "w") as w:
                w.write(r.read().format(scriptdir=scriptdir))
        data_config_train = {
            'reso': (self.opt.resolution, self.opt.resolution)}
        traindata = VideoData(csv_file=csv_file,
                              data_config=data_config_train, transform=None)
        self.train_loader = torch.utils.data.DataLoader(
            traindata, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.batch_size, collate_fn=_collate_filter_none)

        netB = ResnetConditionHR(input_nc=(
            3, 3, 1, 4), output_nc=4, n_blocks1=self.opt.n_blocks1, n_blocks2=self.opt.n_blocks2)
        if self.device == 'cuda':
            netB.cuda()
        netB.eval()
        for param in netB.parameters():  # freeze netB
            param.requires_grad = False
        self.netB = netB

        netG = ResnetConditionHR(input_nc=(
            3, 3, 1, 4), output_nc=4, n_blocks1=self.opt.n_blocks1, n_blocks2=self.opt.n_blocks2)
        netG.apply(conv_init)
        self.netG = netG

        if self.device == 'cuda':
            self.netG.cuda()
            # TODO(asuhan): is this needed?
            torch.backends.cudnn.benchmark = True

        netD = MultiscaleDiscriminator(
            input_nc=3, num_D=1, norm_layer=nn.InstanceNorm2d, ndf=64)
        netD.apply(conv_init)
        netD = nn.DataParallel(netD)
        self.netD = netD
        if self.device == 'cuda':
            self.netD.cuda()

        self.l1_loss = alpha_loss()
        self.c_loss = compose_loss()
        self.g_loss = alpha_gradient_loss()
        self.GAN_loss = GANloss()

        self.optimizerG = optim.Adam(netG.parameters(), lr=1e-4)
        self.optimizerD = optim.Adam(netD.parameters(), lr=1e-5)

        self.log_writer = SummaryWriter(scriptdir)
        self.model_dir = scriptdir

        self._maybe_trace()

    def _maybe_trace(self):
        for data in self.train_loader:
            bg, image, seg, multi_fr = data['bg'], data['image'], data['seg'], data['multi_fr']
            if self.device == 'cuda':
                bg, image, seg, multi_fr = Variable(bg.cuda()), Variable(
                    image.cuda()), Variable(seg.cuda()), Variable(multi_fr.cuda())
            else:
                bg, image, seg, multi_fr = Variable(bg), Variable(
                    image), Variable(seg), Variable(multi_fr)
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
        raise NotImplementedError()

    def train(self, niterations=1):
        self.netG.train()
        self.netD.train()
        lG, lD, GenL, DisL_r, DisL_f, alL, fgL, compL, elapse_run, elapse = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        t0 = time.time()
        KK = len(self.train_loader)
        wt = 1
        epoch = 0
        step = 50

        for i, data in enumerate(self.train_loader):
            if (i > niterations):
                break
            # Initiating

            bg, image, seg, multi_fr, seg_gt, back_rnd = data['bg'], data[
                'image'], data['seg'], data['multi_fr'], data['seg-gt'], data['back-rnd']

            if self.device == 'cuda':
                bg, image, seg, multi_fr, seg_gt, back_rnd = Variable(bg.cuda()), Variable(image.cuda()), Variable(
                    seg.cuda()), Variable(multi_fr.cuda()), Variable(seg_gt.cuda()), Variable(back_rnd.cuda())
                mask0 = Variable(torch.ones(seg.shape).cuda())
            else:
                bg, image, seg, multi_fr, seg_gt, back_rnd = Variable(bg), Variable(
                    image), Variable(seg), Variable(multi_fr), Variable(seg_gt), Variable(back_rnd)
                mask0 = Variable(torch.ones(seg.shape))

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

    def eval(self, niterations=1):
        raise NotImplementedError()
