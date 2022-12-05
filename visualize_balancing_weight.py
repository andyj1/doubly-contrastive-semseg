from PIL import Image
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np

import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np

from utils import utils
from utils.file_io import read_img, read_disp

from pathlib import Path
from dataloaders import custom_transforms as sw
from dataloaders.utils import custom_collate
from dataloaders.datasets import VOCSegmentation, Cityscapes, CityLostFound
from matplotlib import pyplot as plt

def main():
    random_crop_size = 768
    scale = 1
    mean = [73.15, 82.90, 72.3]
    std = [47.67, 48.49, 47.73]
    mean_rgb = tuple(np.uint8(scale * np.array(mean)))
    num_classes = 20

    target_size_crops = (random_crop_size, random_crop_size)
    target_size_crops_feats = (random_crop_size // 4, random_crop_size // 4)
    target_size = (2048, 1024)
    target_size_feats = (2048 // 4, 1024 // 4)


    val_transform = sw.Compose(
        [sw.SetTargetSize(target_size=target_size, target_size_feats=target_size_feats),
         sw.LabelBoundaryTransform(num_classes=num_classes, reduce=True),
         sw.Tensor(),
         ]
    )

    dataset = 'city_lost'
    data_root = os.path.join('/root/dataset', dataset)

    if dataset == 'cityscapes':
        train_dst = Cityscapes(root=data_root, dataset_name=dataset,
                                mode='train', transform=val_transform)
        num_classes = 19
    elif dataset == 'city_lost':
        train_dst = CityLostFound(root=data_root, dataset_name=dataset,
                               mode='train', transform=val_transform)
        num_classes = 20

    train_loader = data.DataLoader(
        train_dst, batch_size=1, shuffle=True, num_workers=4,
        pin_memory=False,
        collate_fn=custom_collate)

    print('Use balanced weights for unbalanced semantic classes...')
    classes_weights_path = os.path.join(data_root,
                                        dataset + '_classes_weights_' +
                                        str(num_classes) + '_new_raw.npy')


    if os.path.isfile(classes_weights_path):
        weight = np.load(classes_weights_path)
    else:
        raise NotImplementedError
    epsilon = 1e-1
    weight = 1 / (np.log(1 + epsilon + weight))
    print(weight)

    if not os.path.exists('paper_fig1'):
        os.mkdir('paper_fig1')

    for img_id, batch in enumerate(train_loader):

        # classes_weights_path = os.path.join(self.opts.data_root,
        #                                     self.opts.dataset + '_classes_weights_19.npy')

        weight_ = weight.astype(np.float32)
        target = np.array(batch['label'][0])
        target[target == 255] = 0
        weight_ = weight_[target]
        # weight_[target == 0] = 0

        label_distance = np.array(batch['label_distance_weight'][0])
        plt.imshow(label_distance, cmap='plasma')
        plt.imsave(os.path.join('paper_fig1', '%d_border_weight.png' % img_id), label_distance)
        # vmax = np.percentile(label_distance, 95)
        # normalizer = mpl.colors.Normalize(vmin=label_distance.min(), vmax=vmax)
        # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        # out_img = (mapper.to_rgba(label_distance)[:, :, :3] * 255).astype(np.uint8)
        # Image.fromarray(out_img).save(os.path.join('paper_fig1', '%d_balance_weight.png' % img_id))

        class_weighted = label_distance * weight_
        plt.imshow(class_weighted, cmap='plasma')
        plt.imsave(os.path.join('paper_fig1', '%d_class_balance_weight.png' % img_id), class_weighted)

        # vmax = np.percentile(class_weighted, 95)
        # normalizer = mpl.colors.Normalize(vmin=class_weighted.min(), vmax=vmax)
        # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        # out_img = (mapper.to_rgba(class_weighted)[:, :, :3] * 255).astype(np.uint8)
        # Image.fromarray(out_img).save(os.path.join('paper_fig1', '%d_class_balance_weight.png' % img_id))


        # image = np.array(batch['left'][0])
        # image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).transpose(1, 2, 0).astype(np.uint8)
        # Image.fromarray(image).show()

        target_ = np.array(batch['label'][0])
        target_ = train_loader.dataset.decode_target(target_).astype(np.uint8)
        Image.fromarray(target_).save(os.path.join('paper_fig1', '%d_gt_sem.png' % img_id))

        # if img_id > 25:
        #     break


main()