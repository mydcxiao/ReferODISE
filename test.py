import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import torchvision

from utils import transforms as T
from utils import utils

import numpy as np
from PIL import Image
import torch.nn.functional as F

import cv2  # type: ignore

from engine import eval_train, eval_test
from build_m import m_model_registry

import json
from typing import Any, Dict, List


def get_dataset(image_set, image_transforms, target_transforms, args, eval_mode=False):
    from res_dataloader import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=image_transforms,
                      target_transforms=target_transforms,
                      eval_mode=eval_mode,
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
    device = torch.device(args.device)
    image_transforms, target_transforms = get_transform(args)
    dataset_test = get_dataset(args.split, image_transforms, target_transforms, args,
                               eval_mode=True)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=args.workers)
    print("Loading model...")
    print(args.model)
    model = m_model_registry[args.model](ck=args.ck)
    model = model.to(device)

    writer = None
    if args.log_dir:
        path = os.path.join(args.log_dir, '{}_{}_test/{}/'.format(args.model, args.dataset,
                                 datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')))
        utils.mkdir(path)
        writer = SummaryWriter(path)

    print("Processing...")

    eval_test(model, data_loader_test, device=device, writer=writer)

    if writer:
        writer.flush()
        writer.close()

    print("Done!")


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
