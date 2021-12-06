"""
YoloV5 Model with DDP support.
Does not support evolving hyperparameters
"""
from subprocess import call
import torch
import random
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from torch.optim import optimizer
from torch.utils.data import dataset

from .yolov5 import val  # for end-of-epoch mAP
from .yolov5.models.yolo import Model
from .yolov5.utils.metrics import fitness
from .yolov5.utils.general import labels_to_image_weights, increment_path, non_max_suppression, scale_coords

random.seed(1337)
torch.manual_seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import os
import math
from pathlib import Path
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION

from .args import parse_opt_train, parse_opt_eval
from .prep import train_prep, eval_prep

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
TRAIN_BATCH_NUM = 1
EVAL_BATCH_NUM = 1

class Model(BenchmarkModel):
    task = COMPUTER_VISION.SEGMENTATION

    def __init__(self, device=None, jit=False, train_bs=1, eval_bs=1):
        train_opt = parse_opt_train()
        # This benchmark does not support evolving hyperparameters
        train_opt.epochs = 1
        train_opt.train_batch_num = TRAIN_BATCH_NUM
        train_opt.evolve = None
        eval_opt = parse_opt_eval()
        eval_opt.eval_batch_num = EVAL_BATCH_NUM

        # setup DDP mode
        self.device = device
        if LOCAL_RANK != -1:
            assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
            assert train_opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
            assert not train_opt.image_weights, '--image-weights argument is not compatible with DDP training'
            assert not train_opt.evolve, '--evolve argument is not compatible with DDP training'
            torch.cuda.set_device(LOCAL_RANK)
            self.device = torch.device('cuda', LOCAL_RANK)
            dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

        # setup other members
        self.train_prep(train_opt.hyp, train_opt, callbacks=None)
        self.eval_prep(eval_opt)

    def eval_prep(self, eval_opt):
        self.eval_opt = eval_opt
        self.eval_dataset, self.eval_webcam, self.eval_model = eval_prep(self.eval_opt)

    def train_prep(self, hyp, opt, callbacks):
        self.train_args = train_prep(hyp, opt, self.device, callbacks)

    def get_module(self):
        pass

    def train(self, niter=1):
        hyp = self.train_args.hyp
        opt = self.train_args.opt
        scaler = self.train_args.scaler
        optimizer = self.train_args.optimizer
        dataset = self.train_args.dataset
        model = self.train_args.train_model
        nb = self.train_args.nb
        nbs = self.train_args.nbs
        maps = self.train_args.maps
        nc = self.train_args.nc
        lf = self.train_args.lf
        gs = self.train_args.gs
        amp = self.train_args.amp
        train_loader = self.train_args.train_loader
        start_epoch = self.train_args.start_epoch
        device = self.train_args.device
        ema = self.train_args.ema
        cuda = self.train_args.cuda
        compute_loss = self.train_args.compute_loss
        batch_size = self.train_args.batch_size
        scheduler = self.train_args.scheduler
        imgsz = self.train_args.imgsz
        noval = self.train_args.opt.noval
        epochs = self.train_args.epochs
        stopper = self.train_args.stopper
        data_dict = self.train_args.data_dict
        single_cls = self.train_args.single_cls
        val_loader = self.train_args.val_loader
        callbacks = self.train_args.callbacks
        best_fitness = self.train_args.best_fitness
        train_batch_num = self.train_args.opt.train_batch_num

        for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
            self.model.train()
            nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
            # Update image weights (optional, single-GPU only)
            if opt.image_weights:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

            # Update mosaic border (optional)
            # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
            # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

            mloss = torch.zeros(3, device=device)  # mean losses
            if RANK != -1:
                train_loader.sampler.set_epoch(epoch)
            pbar = zip(train_batch_num, train_loader)
            # LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
            # if RANK in [-1, 0]:
            #     pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
            optimizer.zero_grad()
            for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

                # Warmup
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                    accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

                # Multi-scale
                if opt.multi_scale:
                    sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Forward
                with amp.autocast(enabled=cuda):
                    pred = model(imgs)  # forward
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                    if RANK != -1:
                        loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                    if opt.quad:
                        loss *= 4.

                # Backward
                scaler.scale(loss).backward()

                # Optimize
                if ni - last_opt_step >= accumulate:
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)
                    last_opt_step = ni

                # Log
                # if RANK in [-1, 0]:
                    # mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                    # mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                    # pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                    #     f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                    # callbacks.run('on_train_batch_end', ni, model, imgs, targets, paths, plots, opt.sync_bn)
                # end batch ------------------------------------------------------------------------------------------------

            # Scheduler
            lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
            scheduler.step()

            if RANK in [-1, 0]:
                # mAP
                # callbacks.run('on_train_epoch_end', epoch=epoch)
                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
                if not noval or final_epoch:  # Calculate mAP
                    results, maps, _ = val.run(data_dict,
                                            batch_size=batch_size // WORLD_SIZE * 2,
                                            imgsz=imgsz,
                                            model=ema.ema,
                                            single_cls=single_cls,
                                            dataloader=val_loader,
                                            save_dir=None,
                                            plots=False,
                                            callbacks=callbacks,
                                            compute_loss=compute_loss)

                # Update best mAP
                fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                if fi > best_fitness:
                    best_fitness = fi
                log_vals = list(mloss) + list(results) + lr
                # callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

                # Save model
                # if (not nosave) or (final_epoch and not evolve):  # if save
                #     ckpt = {'epoch': epoch,
                #             'best_fitness': best_fitness,
                #             'model': deepcopy(de_parallel(model)).half(),
                #             'ema': deepcopy(ema.ema).half(),
                #             'updates': ema.updates,
                #             'optimizer': optimizer.state_dict(),
                #             'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                #             'date': datetime.now().isoformat()}

                #     # Save last, best and delete
                #     torch.save(ckpt, last)
                #     if best_fitness == fi:
                #         torch.save(ckpt, best)
                #     if (epoch > 0) and (opt.save_period > 0) and (epoch % opt.save_period == 0):
                #         torch.save(ckpt, w / f'epoch{epoch}.pt')
                #     del ckpt
                #     callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

                # Stop Single-GPU
                # if RANK == -1 and stopper(epoch=epoch, fitness=fi):
                #     break

                # Stop DDP TODO: known issues shttps://github.com/ultralytics/yolov5/pull/4576
                # stop = stopper(epoch=epoch, fitness=fi)
                # if RANK == 0:
                #    dist.broadcast_object_list([stop], 0)  # broadcast 'stop' to all ranks

            # Stop DPP
            # with torch_distributed_zero_first(RANK):
            # if stop:
            #    break  # must break all DDP ranks
            # end epoch ----------------------------------------------------------------------------------------------------
        # end training -----------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def eval(self, niter=1):
        if self.jit:
            return NotImplementedError("JIT is not supported by this model")
        device = self.device
        dataset = self.eval_dataset
        webcam = self.eval_webcam
        model = self.eval_model
        half = self.eval_opt.half
        augment = self.eval_opt.augment
        max_det = self.eval_opt.max_det
        conf_thres = self.eval_opt.conf_thres
        iou_thres = self.eval_opt.iou_thres
        classes = self.eval_opt.classes
        agnostic_nms = self.eval_opt.agnostic_nms

        dt, seen = [0.0, 0.0, 0.0], 0
        dataset_iter = zip(self.eval_opt.eval_batch_num, dataset)
        for _bactch_num, (path, im, im0s, vid_cap, s) in dataset_iter:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # Inference
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            visualize = False
            pred = model(im, augment=augment, visualize=visualize)

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    _, im0, _ = path[i], im0s[i].copy(), dataset.count
                else:
                    _, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()