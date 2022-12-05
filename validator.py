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
    if opts.dataset == 'cityscapes' or opts.dataset == 'kitti_2015' or opts.dataset == 'kitti_mix':
        opts.num_classes = 19
    elif opts.dataset == 'city_lost':
        opts.num_classes = 20  # 19 cityscapes classes + small obstacle objects
    elif opts.dataset == 'sceneflow':
        opts.num_classes = 0
    else:
        raise NotImplementedError


    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])

    target_size = (2048, 1024)
    target_size_feats = (2048 // 4, 1024 // 4)

    val_transform = sw.Compose(
        [sw.CropBlackArea(),
         sw.SetTargetSize(target_size=target_size, target_size_feats=target_size_feats),
         sw.Tensor(),
         ]
    )

    val_dst = CityLostFound(root=opts.data_root, dataset_name=opts.dataset,
                            mode='val', transform=val_transform)

    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=8,
        pin_memory=False, drop_last=False,
        collate_fn=custom_collate)


    model = network.MyNet_MS(opts,
                             opts.max_disp,
                             no_intermediate_supervision=opts.no_intermediate_supervision,
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

    num_samples = len(val_loader)
    print('=> %d samples found in the test set' % num_samples)

    val_epe, val_d1, val_thres1, val_thres2, val_thres3 = 0, 0, 0, 0, 0
    valid_samples = 0
    score = 0
    mean_epe, mean_d1, mean_thres1, mean_thres2, mean_thres3 = 0, 0, 0, 0, 0
    time_val = []
    time_val_dataloader = []

    data_start = time.time()
    for i, sample in enumerate(val_loader):
        data_time = time.time() - data_start
        time_val_dataloader.append(data_time)

        left = sample['left'].to(device, dtype=torch.float32)
        right = sample['right'].to(device, dtype=torch.float32)

        if 'label' in sample.keys():
            labels = sample['label'].to(device, dtype=torch.long)
        if 'disp' in sample.keys():
            gt_disp = sample['disp'].to(device)
            mask = (gt_disp > 0) & (gt_disp < opts.max_disp)
            if not mask.any():
                continue
        if 'pseudo_disp' in sample.keys():
            pseudo_gt_disp = sample['pseudo_disp'].to(device)
            pseudo_mask = (pseudo_gt_disp > 0) & (pseudo_gt_disp < opts.max_disp) & (~mask)  # inverse mask

        # Warmup
        if i == 0:
            with torch.no_grad():
                for _ in range(10):
                    model(left, right)

        num_imgs += left.size(0)
        valid_samples += 1
        with torch.no_grad():
            start_time = time.time()
            pred_disp_pyramid, left_seg = model(left, right)
            fwt = time.time() - start_time
            time_val.append(fwt)

            pred_disp = pred_disp_pyramid[-1]

            if i % 10 == 0:
                val_epe, val_d1, val_thres1, val_thres2, val_thres3 = \
                    calculate_disparity_error(pred_disp, gt_disp, mask,
                                                   val_epe, val_d1, val_thres1, val_thres2, val_thres3)

            preds = left_seg.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            evaluator.add_batch(targets, preds)

            print("[%d/%d] Model passed time (bath size=%d): %.3f (Mean time per img: %.3f), Dataloader time : %.3f" % (
                i, num_samples,
                opts.val_batch_size, fwt,
                sum(time_val) / len(time_val) / opts.val_batch_size, data_time))

        data_start = time.time()

    score = evaluator.get_results()

    mean_epe = val_epe / valid_samples
    mean_d1 = val_d1 / valid_samples
    mean_thres1 = val_thres1 / valid_samples
    mean_thres2 = val_thres2 / valid_samples
    mean_thres3 = val_thres3 / valid_samples

    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    print(evaluator.to_str(score))

    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {},"
          " epe:{}, d1:{}, thres1:{}, thres2:{} thres3:{}".format(Acc, Acc_class, mIoU, FWIoU,
                                                                  mean_epe, mean_d1,
                                                                  mean_thres1, mean_thres2, mean_thres3))


        # if pred_disp.size(-1) < left.size(-1):
        #     pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
        #     pred_disp = F.interpolate(pred_disp, (left.size(-2), left.size(-1)),
        #                               mode='bilinear') * (left.size(-1) / pred_disp.size(-1))
        #     pred_disp = pred_disp.squeeze(1)  # [B, H, W]


        # save images
        # for b in range(pred_disp.size(0)):
        #     disp = pred_disp[b].detach().cpu().numpy()  # [H, W]
        #     save_name = sample['left_name'][b]
        #     save_name_disp = os.path.join(opts.output_dir, save_name)
        #     check_path(os.path.dirname(save_name_disp))
        #     skimage.io.imsave(save_name_disp, (disp * 256.).astype(np.uint16))
        #
        #     save_name_disp_color = os.path.join(opts.output_dir, 'colorize', save_name)
        #     check_path(os.path.dirname(save_name_disp_color))
        #     # plt.imshow(disp*256., cmap="hsv")
        #     # disp_ = disp[105:,:] * 256.
        #     disp_ = disp * 256.
        #     plt.imshow(disp_)
        #     plt.hsv()
        #     plt.imsave(save_name_disp_color, disp_)


