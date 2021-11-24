import torch
from collections import OrderedDict
from contextlib import suppress
from timm.utils import AverageMeter, reduce_tensor

def train_epoch(
        epoch, model, loader, optimizer, args,
        lr_scheduler=None, saver=None, output_dir='', amp_autocast=suppress, loss_scaler=None, model_ema=None,
        num_batch=1):

    # batch_time_m = AverageMeter()
    # data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    # end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in zip(range(num_batch), loader):
        last_batch = batch_idx == last_idx
        # data_time_m.update(time.time() - end)

        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            output = model(input, target)
        loss = output['loss']

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, clip_grad=args.clip_grad, parameters=model.parameters())
        else:
            loss.backward()
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1

        # batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            # if args.distributed:
            #     reduced_loss = reduce_tensor(loss.data, args.world_size)
            #     losses_m.update(reduced_loss.item(), input.size(0))
            #
            # if args.local_rank == 0:
            #    logging.info(
            #        'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
            #        'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
            #        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
            #        '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
            #        'LR: {lr:.3e}  '
            #        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
            #            epoch,
            #            batch_idx, len(loader),
            #            100. * batch_idx / last_idx,
            #            loss=losses_m,
            #            batch_time=batch_time_m,
            #            rate=input.size(0) * args.world_size / batch_time_m.val,
            #            rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
            #            lr=lr,
            #            data_time=data_time_m))

            #    if args.save_images and output_dir:
            #        torchvision.utils.save_image(
            #            input,
            #            os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
            #            padding=0,
            #            normalize=True)

        # if saver is not None and args.recovery_interval and (
        #        last_batch or (batch_idx + 1) % args.recovery_interval == 0):
        #    saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        # end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])

def validate(model, loader, args, evaluator=None, log_suffix='',
             num_batch=1):
    # batch_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.eval()

    # end = time.time()
    # last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in zip(range(num_batch), loader):
            # last_batch = batch_idx == last_idx

            output = model(input, target)
            loss = output['loss']

            if evaluator is not None:
                evaluator.add_predictions(output['detections'], target)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))

            # batch_time_m.update(time.time() - end)
            # end = time.time()
            # if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
            #     log_name = 'Test' + log_suffix
            #     logging.info(
            #         '{0}: [{1:>4d}/{2}]  '
            #         'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
            #         'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '.format(
            #             log_name, batch_idx, last_idx, batch_time=batch_time_m, loss=losses_m))

    metrics = OrderedDict([('loss', losses_m.avg)])
    if evaluator is not None:
        metrics['map'] = evaluator.evaluate()

    return metrics