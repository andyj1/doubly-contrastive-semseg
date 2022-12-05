from __future__ import absolute_import, division, print_function

from options import Options

options = Options()
opts = options.parse()
from trainer import *


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.nn.parameter import Parameter

from dataloaders.utils import get_dataset, custom_collate
from metrics import Evaluator, TimeAverageMeter

import utils


import random
import numpy as np
import os
import network
from dataloaders.datasets import VOCSegmentation, Cityscapes, CityLostFound, LostFound
from dataloaders import custom_transforms as sw
import skimage.io
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)  # explicitly set exist_ok when multi-processing

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda:{}'.format(opts.gpu_id) if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    n_gpus = len(opts.gpu_id)
    print("Number of used GPU : {}".format(n_gpus))
    print("Used GPU ID : {}".format(opts.gpu_id))

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    torch.backends.cudnn.benchmark = True

    opts.data_root = os.path.join(opts.data_root, opts.dataset)
    opts.num_classes = 19

    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])

    random_crop_size = (896, 256)
    target_size_crops = random_crop_size
    target_size_crops_feats = (random_crop_size[0] // 4, random_crop_size[1] // 4)
    target_size = (1280, 384)
    target_size_feats = (1280 // 4, 384 // 4)


    test_transform = sw.Compose(
        [   #sw.RandomCrop_PIL(384, 1280, validate=True),
            #sw.SetTargetSize(target_size=target_size_crops, target_size_feats=target_size_crops_feats),
            sw.Tensor(),
        ]
    )

    test_dst = Cityscapes(root=opts.data_root, dataset_name=opts.dataset,
                         mode='test', transform=test_transform, opts=opts)

    test_loader = data.DataLoader(
        test_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=4,
        pin_memory=True, drop_last=False,
        collate_fn=custom_collate)

    model = network.RODSNet(opts,
                             opts.max_disp,
                             num_classes=opts.num_classes,
                             device=device,
                             refinement_type=opts.refinement_type,
                             )
    model.to(device)

    evaluator = Evaluator(opts.num_classes)

    if opts.resume is not None:
        if not os.path.isfile(opts.resume):
            raise RuntimeError("=> no checkpoint found at '{}'".format(opts.resume))
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.resume, map_location=device)

        loaded_pt = checkpoint['model_state']
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in loaded_pt.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict, strict=False)
    else:
        print("[!] No checkpoints found, Retrain...")

    num_params = utils.count_parameters(model)
    print('=> Number of trainable parameters: %d' % num_params)

    # Inference
    model.eval()

    inference_time = 0
    num_imgs = 0

    num_samples = len(test_loader)
    print('=> %d samples found in the test set' % num_samples)


    for i, sample in enumerate(tqdm(test_loader)):
        if i % 100 == 0:
            print('=> Inferencing %d/%d' % (i, num_samples))

        left = sample['left'].to(device, dtype=torch.float32)
        right = sample['right'].to(device, dtype=torch.float32)

        # Pad
        ori_height, ori_width = left.size()[2:]
        if ori_height < opts.val_img_height or ori_width < opts.val_img_width:
            top_pad = opts.val_img_height - ori_height
            right_pad = opts.val_img_width - ori_width
            # print(top_pad, right_pad)

            # Pad size: (left_pad, right_pad, top_pad, bottom_pad)
            left = F.pad(left, (0, right_pad, top_pad, 0))
            right = F.pad(right, (0, right_pad, top_pad, 0))

        num_imgs += left.size(0)
        with torch.no_grad():
            time_start = time.time()

            pred_disp_pyramid, left_seg = model(left, right)
            pred_disp = pred_disp_pyramid[-1]

            inference_time += time.time() - time_start


        image = left[0].detach().cpu().numpy()
        right_image = right[0].detach().cpu().numpy()
        print(left_seg.shape)
        pred_sem = left_seg.detach().max(dim=1)[1].cpu().numpy()
        print(pred_sem.shape)

        # Crop
        if ori_height < opts.val_img_height or ori_width < opts.val_img_width:
            if right_pad != 0:
                pred_disp = pred_disp[:, top_pad:, :-right_pad]
                image = image[:, top_pad:, :-right_pad]
                right_image = right_image[:, top_pad:, :-right_pad]
                pred_sem = pred_sem[:, top_pad:, :-right_pad]

            else:
                pred_disp = pred_disp[:, top_pad:]
                image = image[:, top_pad:]
                right_image = right_image[:, top_pad:]
                pred_sem = pred_sem[:, top_pad:]

        for b in range(pred_disp.size(0)):
            disp = pred_disp[b].detach().cpu().numpy()  # [H, W]
            save_name = sample['left_name'][b]
            save_name_disp = os.path.join(opts.output_dir, save_name)
            check_path(os.path.dirname(save_name_disp))
            skimage.io.imsave(save_name_disp, (disp * 256.).astype(np.uint16))

            save_name_disp_color = os.path.join(opts.output_dir, 'colorize', save_name)
            check_path(os.path.dirname(save_name_disp_color))
            # plt.imshow(disp*256., cmap="hsv")
            # disp_ = disp[105:,:] * 256.
            disp_ = disp * 256.
            plt.imshow(disp_)
            plt.viridis()
            plt.imsave(save_name_disp_color, disp_)

            # image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
            image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).transpose(1, 2, 0).astype(np.uint8)
            save_name_left = os.path.join(opts.output_dir, 'left', save_name)
            check_path(os.path.dirname(save_name_left))
            image = Image.fromarray(image)
            image.save(save_name_left)

    print('=> Mean inference time for %d images: %.3fs' % (num_imgs, inference_time / num_imgs))