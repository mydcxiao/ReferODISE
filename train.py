import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision

from functools import reduce
import operator

from utils import transforms as T
from utils import utils
import numpy as np

import torch.nn.functional as F

import gc
from collections import OrderedDict

from criterion import Criterion
from engine import train_one_epoch, eval_train
from build_m import m_model_registry


def get_dataset(image_set, image_transforms, target_transforms, args):
    from res_dataloader import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=image_transforms,
                      target_transforms=target_transforms,
                      )
    return ds

def get_transform(args):
    image_transforms = [T.Resize(args.img_size, args.img_size),
                        T.ToTensor(),
                        ]
    target_transforms = [T.Resize(args.img_size // 4, args.img_size // 4),
                         T.ToTensor(),
                         ]
    
    return T.Compose(image_transforms), T.Compose(target_transforms)


def main(args):
    image_transforms, target_transforms = get_transform(args)

    dataset = get_dataset(
                          args.split,
                          image_transforms=image_transforms,
                          target_transforms=target_transforms,
                          args=args
                          )
    dataset_test = get_dataset(
                               "val",
                               image_transforms=image_transforms,
                               target_transforms=target_transforms,
                               args=args,
                               )

    # batch sampler
    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                                    shuffle=True)
    

    # test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, num_replicas=num_tasks, rank=global_rank,
                                                                   shuffle=False)

    # data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)

    # data_loader_test = torch.utils.data.DataLoader(
    #     dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, 
        sampler=test_sampler, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)

    # model initialization
    print(args.model)
    model = m_model_registry[args.model](ck=args.ck,
                                         ck_image_encoder=args.ck_image_encoder,
                                         ck_pixel_decoder=args.ck_pixel_decoder,
                                         ck_prompt_encoder=args.ck_prompt_encoder,
                                         ck_mask_decoder=args.ck_mask_decoder)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    single_model = model.module

    # criterion for training 
    criterion = Criterion(args.batch_size, 
                          args.coef_focal_loss, 
                          args.coef_dice_loss,
    )
    criterion = torch.nn.SyncBatchNorm.convert_sync_batchnorm(criterion)
    criterion.cuda()
    # criterion = torch.nn.parallel.DistributedDataParallel(criterion, device_ids=[args.local_rank], find_unused_parameters=True)
    criterion = torch.nn.parallel.DistributedDataParallel(criterion, device_ids=[args.local_rank])


    # resume training
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'], strict=False)

    # parameters to optimize
    # layer-wise learning rate decay
    # layer_ld = args.layer_ld
    no_decay = ["bias", "norm", 'mask_tokens', 'positional_embedding', 'pe_layer']
    params = []
    params_no_decay = []
    for name, param in single_model.named_parameters():
        if ('prompt_encoder' in name or 'mask_decoder' in name) \
        and 'prompt_encoder.clip' not in name and param.requires_grad:
            if not any(nd in name for nd in no_decay):
                params.append(param)
            else:
                params_no_decay.append(param)

    params_to_optimize = [
        {"params": params, "lr": args.lr},
        {"params": params_no_decay, "lr": args.lr, "weight_decay": 0.0},
    ]   

    # optimizer
    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad
                                  )

    # learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
    #                                                 lambda x: x / args.warmup if x < args.warmup else \
    #                                                 1 - (0.8 ** (x // len(data_loader)) * (1 - 0.8) / (len(data_loader) - args.warmup) * (x - args.warmup)) if x < len(data_loader) else \
    #                                                 0.8 ** (x // len(data_loader)) - (0.8 ** (x // len(data_loader)) * (1 - 0.8) / len(data_loader) * (x - x // len(data_loader) * len(data_loader))) if x % len(data_loader) != 0 else \
    #                                                 0.8 ** (x // len(data_loader))
    #                                                 ) 

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(data_loader), eta_min=args.lr_min)

    # lr_scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: x / args.warmup if x < args.warmup else 1)
    # lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(data_loader) - args.warmup, eta_min=args.lr_min)
    # lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [lr_scheduler1, lr_scheduler2], milestones=[args.warmup])
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    # housekeeping
    start_time = time.time()
    iterations = 0
    best_oIoU = -0.1

    # resume training (optimizer, lr scheduler, and the epoch)
    resume_epoch = -999
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
        # data_loader.dataset.resume(checkpoint['last_pred'])
    else:
        resume_epoch = -999

    # SummaryWriter
    writer = None
    path = ''
    if args.resume:
        path = checkpoint['writer']
        if path:
            writer = SummaryWriter(path)
    elif args.log_dir:
        path = os.path.join(args.log_dir, '{}_{}_train/{}/'.format(args.model, args.dataset,
                                 datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')))
        utils.mkdir(path)
        writer = SummaryWriter(path)
    
    #training loops 
    for epoch in range(max(0, resume_epoch+1), args.epochs):
        data_loader.sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, args.print_freq,
                    iterations, writer)
        # data_loader.sampler.dataset.update_last_pred()
        
        # if epoch % 5 == 0:
        data_loader_test.sampler.set_epoch(epoch)
        iou, overallIoU = eval_train(model, data_loader_test, epoch, criterion, writer)

        print('Average object IoU {}'.format(iou))
        print('Overall IoU {}'.format(overallIoU))
        save_checkpoint = (best_oIoU < overallIoU)
        if save_checkpoint:
            print('Better epoch: {}\n'.format(epoch))

            dict_to_save = {'model': single_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch, 
                            'args': args,
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'writer': path,
                            # 'last_pred': data_loader.dataset.last_pred,
                            }

            utils.save_on_master(dict_to_save, os.path.join(args.ck_dir,
                                                            '{}_best_{}_{}.pth'.format(args.model, args.dataset, args.splitBy)))

            best_oIoU = overallIoU

    # summarize
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if writer:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    # set up distributed learning
    utils.init_distributed_mode(args)
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
