import torch
import torch.nn as nn
from torch.utils import data
from torch.nn.parameter import Parameter

from dataloaders.utils import get_dataset, custom_collate
from metrics import Evaluator, TimeAverageMeter

import utils
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.calculate_weights import calculate_weigths_labels_new
from utils.loss import BoundaryAwareFocalLoss, FocalLoss2, SupConLoss, PixelContrastLoss

import random
import numpy as np
import os

import network

class InitOpts():
    def __init__(self, options):
        self._init_options(options)
        self._init_gpu_settings()
        self._init_saver()
        self._init_dataloader() # needs to be after saver (exp_dir path), criterion (weight)
        self._init_criterion()
        self._init_model_map()
        self._init_optimizer()
        self._init_checkpoints()
        self._init_scheduler()

        # self._init_multi_gpu_setting()
        if not self.opts.test_only:
            self._init_tensorboard_writer()

    def _init_options(self, options):
        self.opts = options

        if self.opts.dataset == 'cityscapes' or self.opts.dataset == 'kitti_2015'\
                or self.opts.dataset == 'kitti_mix' or self.opts.dataset == 'acdc' or self.opts.dataset == 'acdc_city':
            self.opts.num_classes = 19
        elif self.opts.dataset == 'city_lost':
            self.opts.num_classes = 20      # 19 cityscapes classes + small obstacle objects
        elif self.opts.dataset == 'sceneflow':
            self.opts.num_classes = 0
        else:
            raise NotImplementedError

        # set data path using opts.dataset name
        self.opts.data_root = os.path.join(self.opts.data_root, self.opts.dataset)

        self.batch_time_t = TimeAverageMeter()
        self.data_time_t = TimeAverageMeter()
        self.batch_time_e = TimeAverageMeter()
        self.time_val = []
        self.time_val_dataloader = []

    def _init_gpu_settings(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.opts.gpu_id
        self.device = torch.device('cuda:{}'.format(self.opts.gpu_id) if torch.cuda.is_available() else 'cpu')
        print("Device: %s" % self.device)

        self.n_gpus = len(self.opts.gpu_id)
        print("Number of used GPU : {}".format(self.n_gpus))
        print("Used GPU ID : {}".format(self.opts.gpu_id))

        # Setup random seed
        torch.manual_seed(self.opts.random_seed)
        np.random.seed(self.opts.random_seed)
        random.seed(self.opts.random_seed)
        torch.backends.cudnn.benchmark = False

        torch.cuda.manual_seed(self.opts.random_seed)
        torch.cuda.manual_seed_all(self.opts.random_seed)
        torch.backends.cudnn.determinisitc = True

    def _init_dataloader(self):
        # Setup dataloader
        self.denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        if self.opts.test_only:
            self.opts.val_batch_size = 1

        self.train_dst, self.val_dst = get_dataset(self.opts)
        self.train_loader = data.DataLoader(
            self.train_dst, batch_size=self.opts.batch_size, shuffle=True, num_workers=4,
            pin_memory=False,
            drop_last=True, collate_fn=custom_collate)
        self.val_loader = data.DataLoader(
            self.val_dst, batch_size=self.opts.val_batch_size, shuffle=False, num_workers=4,
            pin_memory=False,
            collate_fn=custom_collate)
        print("Dataset: %s, Train set: %d, Val set: %d" %
              (self.opts.dataset, len(self.train_dst), len(self.val_dst)))

    def _init_model_map(self):
        # Set up model        
        if self.opts.deeplab or self.opts.model == 'enet':
            from network import modeling
            # load models: ['enet', 'deeplabv3plus_resnet50', ...]
            self.model = modeling.__dict__[self.opts.model](self.opts, num_classes=self.opts.num_classes, output_stride=self.opts.output_stride)
            
        else:
            self.model = network.WeatherNet(self.opts,
                                    num_classes=self.opts.num_classes,
                                    device=self.device,
                                    backbone=self.opts.model,
                                    train_semantic=self.opts.train_semantic
                                    )
        self.weather_clf = network.WeatherClassifier(self.opts, weather_class_num=self.opts.weather_num)

        utils.count_parameters(self.model, self.opts)

        # import importlib
        # torchsummary_found = importlib.util.find_spec("torchsummary")
        # if torchsummary_found is not None:
        #     from torchsummary import summary
        #     print(summary(self.model.to(self.device), (3, self.opts.crop_size, self.opts.crop_size)))
        # import sys; sys.exit(1)

    def _init_optimizer(self):
        # Set up metrics
        self.evaluator = Evaluator(self.opts.num_classes, self.opts.weather_num)

        # Set up optimizer
        if self.opts.optimizer_policy == 'SGD':
            specific_params = list(filter(utils.filter_specific_params,
                                          self.model.named_parameters()))
            base_params = list(filter(utils.filter_base_params,
                                      self.model.named_parameters()))

            feature_extractor_params = list(filter(utils.filter_feature_extractor_params,
                                                   self.model.named_parameters()))

            specific_params = [kv[1] for kv in specific_params]  # kv is a tuple (key, value)
            base_params = [kv[1] for kv in base_params]
            feature_extractor_params = [kv[1] for kv in feature_extractor_params]
            specific_lr = self.opts.lr * 0.1
            feature_extractor_lr = self.opts.lr

            params_group = [
                {'params': base_params, 'lr': self.opts.lr},
                {'params': specific_params, 'lr': specific_lr},
                {'params': feature_extractor_params, 'lr': feature_extractor_lr},
            ]

            if self.opts.train_semantic:
                semantic_params = list(filter(utils.filter_semantic_params,
                                              self.model.named_parameters()))
                semantic_params = [kv[1] for kv in semantic_params]
                semantic_lr = self.opts.lr * 10

                params_group += [
                    {'params': semantic_params, 'lr': semantic_lr},
                ]

            self.optimizer = torch.optim.SGD(params_group,
                                             lr=self.opts.lr, momentum=0.9, weight_decay=self.opts.weight_decay)
        elif self.opts.optimizer_policy == 'ADAM':
            # Optimizer

            if self.opts.deeplab or self.opts.model == 'enet':
                train_params = [
                    {'params': self.model.parameters(), 'lr': self.opts.lr,
                    'weight_decay': self.opts.weight_decay},
                ]
            else:
                fine_tune_factor = 4
                train_params = [
                    {'params': self.model.random_init_params(), 'lr': self.opts.lr,
                    'weight_decay': self.opts.weight_decay},
                    {'params': self.model.fine_tune_params(), 'lr': self.opts.lr / fine_tune_factor,
                    'weight_decay': self.opts.weight_decay / fine_tune_factor},
                ]
            
            self.optimizer = torch.optim.Adam(train_params, betas=(0.9, 0.99))

        else:
            raise NotImplementedError

    def _init_criterion(self):
        ''' Set up criterion '''
        # whether to use class balanced weights
        if self.opts.use_balanced_weights and self.opts.train_semantic:
            # print('Use balanced weights for unbalanced semantic classes...')
            # classes_weights_path = os.path.join(self.opts.data_root,
            #                                     self.opts.dataset + '_classes_weights_19.npy')
            if self.opts.dataset == "acdc_city":
                print(self.opts.data_root.replace('acdc_city', 'acdc'))
                classes_weights_path = os.path.join(self.opts.data_root.replace('acdc_city', 'acdc'),
                                                    self.opts.dataset + '_classes_weights_' +
                                                    str(self.opts.num_classes) + '_new_raw.npy')
            else:
                classes_weights_path = os.path.join(self.opts.data_root,
                                                    self.opts.dataset + '_classes_weights_' +
                                                    str(self.opts.num_classes) + '_new_raw.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels_new(classes_weights_path, self.opts.dataset, self.train_loader, self.opts.num_classes)

            print('Whole_datasets pixel ratio: {}'.format(weight))
            epsilon = self.opts.epsilon  # experimental setup
            weight = 1 / (np.log(1 + epsilon + weight))
            print('refined pixel ratio for class imbalance: {}'.format(weight))
            print('max/min ratio: {}'.format(np.max(weight)/np.min(weight)))
            weight = torch.from_numpy(weight.astype(np.float32))
            print('using weights...')
        else:
            weight = None

        self.opts.weight = weight

        if self.opts.train_semantic:
            self.criterion = BoundaryAwareFocalLoss(gamma=0.5, num_classes=self.opts.num_classes,
															ignore_id=255, weight=weight, device=self.device,
															opts=self.opts)
            self.focal_criterion = FocalLoss2(gamma=.5, num_classes=self.opts.num_classes,
                                        ignore_id=255, weight=weight, device=self.device)
            self.supcon_criterion = SupConLoss(temperature=0.07, contrast_mode='all', base_temperature=0.07, weight=weight, device=self.device, opts=self.opts)
            self.pixelcontrast_criterion = PixelContrastLoss(device=self.device)
            self.ce_criterion = nn.CrossEntropyLoss(weight=None, ignore_index=255) # not weighted
            
        # disparity Loss weights
        self.pyramid_weight = {
            5: [1 / 3, 2 / 3, 1.0, 1.0, 1.0],
            4: [1 / 3, 2 / 3, 1.0, 1.0],
            3: [1.0, 1.0, 1.0],     # 1 scale only
            2: [1.0, 1.0],
            1: [1.0]                # highest loss only
        }

    def _init_checkpoints(self):
        # Restore
        self.cur_epochs = 0
        self.num_iter = 0

        self.best_score = 0.0
        self.best_score_epoch = -1

        if self.opts.resume is not None:
            if not os.path.isfile(self.opts.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(self.opts.resume))
            # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
            checkpoint = torch.load(self.opts.resume, map_location=self.device)
            self._init_multi_gpu_setting()
            loaded_pt = checkpoint['model_state']
            model_dict = self.model.state_dict()

            if self.opts.continue_training:
                self.opts.start_epoch = checkpoint['epoch'] + 1
                self.cur_epochs = checkpoint['epoch'] + 1
                self.num_iter = checkpoint['num_iter'] + 1
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                self.best_score = checkpoint['best_score']
                self.best_score_epoch = checkpoint['best_score_epoch']
                
                # 1. filter out unnecessary keys
                pretrained_dict = {k: v for k, v in loaded_pt.items() if k in model_dict}
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                # 3. load the new state dict
                self.model.load_state_dict(model_dict, strict=False)
                print("Training state restored from %s" % self.opts.resume)
                print("=> loaded checkpoint '{}' (epoch {})".format(self.opts.resume, checkpoint['epoch']))
                print("Resume Training from epochs {}".format(self.cur_epochs))

            else:
            # elif self.opts.finetuning:
                # print("Just Testing results of pretrained network model...")
                #
                print("If you want to continue training with checkpoints, add --continue_training options!")
                # 1. filter out unnecessary keys
                pretrained_dict = {k: v for k, v in loaded_pt.items() if k in model_dict}
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                # 3. load the new state dict
                self.model.load_state_dict(model_dict, strict=False)

            del checkpoint  # free memory
        else:
            print("[!] No checkpoints found, Retrain...")
            # self._init_multi_gpu_setting()

        # if self.opts.dataset == 'kitti_mix':
        #     # freeze semantic modules
        #     print("[Freeze semantic segmentation modules in kitti_mix...]")
        #     for name, p in self.model.named_parameters():
        #         # "feature_extractor.upsample_blends."
        #         if "feature_extractor.upsample_blends." in name:
        #             print(name)
        #             p.requires_grad = False
        #         if "segmentation." in name:
        #             print(name)
        #             p.requires_grad = False


    def _init_scheduler(self):
        # lr_min = 1e-6
        lr_min = self.opts.last_lr
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.opts.epochs, lr_min)
        if self.opts.continue_training:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.opts.epochs, lr_min, last_epoch=self.cur_epochs)
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.opts.epochs, lr_min)



    def _init_multi_gpu_setting(self):
        if self.n_gpus > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

    def _init_saver(self):
        # Define Saver
        self.saver = Saver(self.opts)
        self.saver.save_experiment_config()
        self.opts.experiment_dir = self.saver.experiment_dir

    def _init_tensorboard_writer(self):
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()