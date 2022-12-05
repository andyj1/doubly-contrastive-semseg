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

class ACDC(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('unlabeled',            0,  255,    'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1,  255,    'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2,  255,    'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3,  255,    'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4,  255,    'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5,  255,    'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6,  255,    'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7,  0,      'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8,  1,      'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9,  255,    'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255,    'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2,      'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3,      'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4,      'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255,    'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255,    'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255,    'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5,      'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255,    'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6,      'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7,      'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8,      'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9,      'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10,     'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11,     'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12,     'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13,     'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14,     'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15,     'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255,    'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255,    'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16,     'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17,     'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18,     'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, 255,     'vehicle', 7, False, True, (0, 0, 142)), # -1 --> 255
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    # color_to_eval_id = dict(zip([c.color for c in classes], [c.train_id for c in classes]))
    color_to_eval_id = { c.color : c.train_id for c in classes }
    # print(color_to_train_id)

    def __init__(self, root: Path, dataset_name='cityscapes', mode='test', save_filename=False,
                 transform=None, opts=None):
        self.root = root
        self.mode = mode
        self.save_filename = save_filename
        self.transform = transform
        self.dataset_name = dataset_name
        self.ignore_index = 255
        self.opts = opts
        
        acdc_dict = None
        if self.opts.debug:
            acdc_dict = {
                'train': 'filenames/acdc/acdc_train_small.txt',
                'val': 'filenames/acdc/acdc_val_small.txt',
                'test': 'filenames/acdc/acdc_test_small.txt'
            }
        else:
            acdc_dict = {
                'train': 'filenames/acdc/acdc_train.txt',
                'val': 'filenames/acdc/acdc_val.txt',
                'test': 'filenames/acdc/acdc_test.txt'
            }

        self.weather_dict = {
            'fog': 0,
            'night': 1,
            'rain': 2,
            'snow': 3,
            # 'sunny': 4
        }

        if mode not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        self.samples = []
        dataset_dict = {'acdc': acdc_dict}
        data_filenames = dataset_dict[dataset_name][mode]
        lines = utils.read_text_lines(data_filenames)
        
        import logging
        for line in lines:
            splits = line.split()

            left_img = splits[0]
            gt_weather = splits[1]
            gt_label = None if len(splits) == 2 else splits[2]
            
            # filter by weather conditions
            if self.opts.weather_condition is not None:
                if gt_weather != self.opts.weather_condition:
                    # logging.info(f"[DATA] Skipping file with weather condition '{gt_weather}'\n")
                    continue
                else:
                    sample = dict()

                    sample['left_name'] = left_img.split('/', 1)[1]
                    
                    # if opts.use_test_data:
                    sample['frame_name'] = left_img.split('/')[5].replace('_rgb_anon', '*') 
                    
                    sample['left'] = os.path.join(self.root, left_img)

                    sample['weather'] = self.weather_dict[gt_weather]
                    sample['label'] = os.path.join(self.root, gt_label) if gt_label is not None else None

                    sample['disp'] = None
                    sample['pseudo_disp'] = None

                    self.samples.append(sample)            
            else:
                sample = dict()

                sample['left_name'] = left_img.split('/', 1)[1]
                
                # if opts.use_test_data:
                sample['frame_name'] = left_img.split('/')[5].replace('_rgb_anon', '*') 
                
                sample['left'] = os.path.join(self.root, left_img)

                sample['weather'] = self.weather_dict[gt_weather]
                sample['label'] = os.path.join(self.root, gt_label) if gt_label is not None else None

                sample['disp'] = None
                sample['pseudo_disp'] = None

                self.samples.append(sample)
        # print(''.join([f'{line}\n' for line in lines]))
        

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target).astype('uint8')]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]
    
    @classmethod
    def convert_color_to_eval_id(cls, pixel_rgb):
        trainId = cls.color_to_eval_id[pixel_rgb]
        if trainId == 19: trainId = 255
        return trainId

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        sample = {}
        sample_path = self.samples[index]
        
        sample['left'] = read_img(sample_path['left'])  # [H, W, 3]
        original_left_image = sample['left']
        #sample['right'] = read_img(sample_path['right'])

        sample['left_name'] = sample_path['left_name']
        #sample['right_name'] = sample_path['right_name']
        
        # if self.opts.use_test_data:
        sample['frame_name'] = sample_path['frame_name']

        # GT disparity of subset if negative, finalpass and cleanpass is positive
        subset = True if 'subset' in self.dataset_name else False
        if sample_path['disp'] is not None:
            sample['disp'] = read_disp(sample_path['disp'], subset=subset)  # [H, W]

        if sample_path['pseudo_disp'] is not None:
            sample['pseudo_disp'] = read_disp(sample_path['pseudo_disp'], subset=subset)  # [H, W]

        if sample_path['label'] is not None:
            label = self.encode_target(Image.open(sample_path['label']))
            sample['label'] = label
            sample['label'] = Image.fromarray(label.astype('uint8'))
            
            # temp = np.array(sample['label']).reshape(1,-1)
            # import logging
            # logging.info(f"gt label id classes: {np.unique(temp)}, {len(np.unique(temp))}")
            # import sys; sys.exit(1)

        sample['weather'] = np.array([sample_path['weather']])

        if self.transform is not None:
            sample = self.transform(sample)

        
        if self.opts.viz_EDT:
            import skimage.io
            from torchvision.transforms import ToPILImage
            import matplotlib.pyplot as plt

            # save original left image
            image_save_dir = os.path.join(self.opts.experiment_dir, 'transformed_images')
            utils.mkdir(image_save_dir)
            save_path = os.path.join(image_save_dir, f'{index}_RGB_orig')
            original_left_image.save(save_path+'.png')

            # save transformed left image
            save_path = os.path.join(image_save_dir, f'{index}_RGB')
            left_image = ToPILImage()(sample['left'].detach().cpu())
            left_image.save(save_path+'.png')

            # save transformed images
            utils.mkdir(image_save_dir)
            EDT = sample['label_distance_weight'] # [H, W]
            save_path = os.path.join(image_save_dir, f'{index}_EDT')
            skimage.io.imsave(save_path+'.png', (EDT * 256.).astype(np.uint16))
            
            # save EDT
            EDT_colored = EDT * 256.
            plt.imshow(EDT_colored)
            plt.viridis()
            plt.imsave(save_path+'_colored'+'.png', EDT_colored)
            plt.clf()

            # save corresponding label map
            target_temp = sample['label'].clone()
            target_temp[target_temp == 255] = 0
            EDT_weighted_colored = EDT * self.opts.weight[target_temp].detach().cpu().numpy() * 256.
            plt.imshow(EDT_weighted_colored)
            plt.viridis()
            plt.imsave(save_path+'_colored_weighted'+'.png', EDT_weighted_colored)
            plt.clf()
            
        return sample

    def __len__(self):
        return len(self.samples)


class Relabel(object):
    def __init__(self, olabel, nlabel):  # change trainid label from olabel to nlabel
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        # assert (isinstance(tensor, torch.LongTensor) or isinstance(tensor,
        #                                                            torch.ByteTensor)), 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor