from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.backbone.efficientnet_pyramid import efficientnet_pyramid
from network.backbone.resnet_pyramid import resnet18_pyramid, resnet34_pyramid
from network.classifier import WeatherClassifier
from network.utils import _BNReluConv, upsample

# from network.backbone.mobilenetv2_pyramid import mobilenetv2_pyramid

class WeatherNet(nn.Module):
    def __init__(self,
                 opts,
                 num_downsample=2,
                 num_classes=19,
                 device=None,
                 feature_similarity='correlation',
                 aggregation_type='adaptive',
                 num_scales=3,
                 backbone='resnet34',
                 train_semantic=True):
        super(WeatherNet, self).__init__()

        self.opts = opts
        # self.refinement_type = refinement_type
        self.num_downsample = num_downsample
        self.aggregation_type = aggregation_type
        self.num_scales = num_scales
        #self.max_disp = max_disp // 4      # it depends on feature extractor's width-height scale
        self.num_classes = num_classes
        self.device = device

        scale = 1
        mean = [73.15, 82.90, 72.3]
        std = [47.67, 48.49, 47.73]

        if backbone == 'resnet18':
            self.feature_extractor = resnet18_pyramid(pretrained=True, pyramid_levels=3, k_upsample=3,
                                                    scale=scale, mean=mean, std=std, k_bneck=1, output_stride=4,
                                                    efficient=True)

        elif backbone == 'resnet34':
            self.feature_extractor = resnet34_pyramid(pretrained=True, pyramid_levels=3, k_upsample=3,
                                                    scale=scale, mean=mean, std=std, k_bneck=1, output_stride=4,
                                                    efficient=True)

        elif backbone == 'efficientnetb0':
            self.feature_extractor = efficientnet_pyramid(pretrained=True, pyramid_levels=3, k_upsample=3,
                                                          mean=mean, std=std, num_classes=self.num_classes,
                                                          )
        # elif backbone == 'mobilenetv2':
        #     self.feature_extractor = mobilenetv2_pyramid(pretrained=True
        #                                                  )
        else:
            raise NotImplementedError

        if train_semantic:
            self.segmentation = _BNReluConv(self.feature_extractor.num_features, 
                                            self.num_classes, batch_norm=True, k=1, bias=True)
            self.loss_ret_additional = False
            self.img_req_grad = False
            self.upsample_logits = True
            self.multiscale_factors = (.5, .75, 1.5, 2.)

    def feature_extraction(self, img):
        x_sem, additional = self.feature_extractor(img)
        return x_sem, additional

    def predict_segmentation(self, features):
        segmentation = self.segmentation.forward(features)
        return segmentation

    def forward(self, left_img, return_supcon_feature=False):
        fine_feat, coarse_feat = self.feature_extraction(left_img) # [bsz*2, *] if supcon
        if return_supcon_feature:
            bsz = fine_feat.shape[0]//2
            fine_feat0 = torch.split(fine_feat, [bsz, bsz], dim=0)[0] # [bsz, *]
        else:
            fine_feat0 = fine_feat # [bsz, *]

        pred_segmap_beforeup = self.predict_segmentation(fine_feat0)
        pred_segmap = upsample(pred_segmap_beforeup, left_img.shape[2:])

        # pred_weather = self.weather_cls(additional['skips_0'])    --> weather classification accuracy slowly decreases...

        # to return coarse feature for 'supcon' type of loss
        # if self.opts.coarse_features:
        #     fine_feat = coarse_feat['skips_0']  # [BSZ (*2 if supcon), C (128), H (6), W (6)]
            # print(f'coarse features shape: {fine_feat.shape}')
            # import sys; sys.exit(1)
        
        # print(f'fine features shape: {fine_feat.shape}')
        # import sys; sys.exit(1)

        return pred_segmap, pred_segmap_beforeup, fine_feat, fine_feat0

    def random_init_params(self):
        # return chain(*([self.segmentation.parameters(), self.feature_extractor.random_init_params()]))
        return self.feature_extractor.random_init_params()

    def fine_tune_params(self):
        return self.feature_extractor.fine_tune_params()