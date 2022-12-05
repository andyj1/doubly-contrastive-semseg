import torch.nn as nn
import torch.nn.functional as F
import torch

upsample = lambda x, size: F.interpolate(x, size, mode='bilinear', align_corners=False)
class SemsegCrossEntropy(nn.Module):
    def __init__(self, num_classes=19, ignore_id=255, print_each=20):
        super(SemsegCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.ignore_id = ignore_id
        self.step_counter = 0
        self.print_each = print_each

    def loss(self, y, t):
        if y.shape[2:4] != t.shape[1:3]:
            y = upsample(y, t.shape[1:3])
        return F.cross_entropy(y, target=t, ignore_index=self.ignore_id)

    def forward(self, logits, labels, **kwargs):
        loss = self.loss(logits, labels)
        # if (self.step_counter % self.print_each) == 0:
        #     print(f'Step: {self.step_counter} Loss: {loss.data.cpu().item():.4f}')
        self.step_counter += 1
        return loss


class BoundaryAwareFocalLoss(nn.Module):
    def __init__(self, gamma=0, num_classes=19, ignore_id=19, print_each=20, weight=None, device=None, opts=None):
        super(BoundaryAwareFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_id = ignore_id
        self.print_each = print_each
        self.step_counter = 0
        self.gamma = gamma
        self.weight = weight
        self.device = device
        self.opts = opts

    def forward(self, input, target, batch, **kwargs):
        # print(input.shape, target.shape)
        if input.shape[-2:] != target.shape[-2:]:
            input = upsample(input, target.shape[-2:])
        target[target == self.ignore_id] = 0  # we can do this because alphas are zero in ignore_id places
        label_distance_weight = batch['label_distance_weight'].to(self.device)
        N = (label_distance_weight.data > 0.).sum()
        if N.le(0):
            return torch.zeros(size=(0,), device=self.device, requires_grad=True).sum()
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        alphas = label_distance_weight.view(-1)
        weight = self.weight[target].view(-1).to(self.device)

        if self.opts.with_depth_level_loss:
            disp_distance_weight = batch['disp_distance_weight'].to(self.device)
            betas = disp_distance_weight.view(-1)

        logpt = F.log_softmax(input.to(torch.float32), dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.detach().exp()

        if self.opts.criterion == 'plain_focal':
            loss = -1 * torch.exp(self.gamma * (1 - pt)) * logpt
        elif self.opts.no_class_weights:
            loss = -1 * alphas * torch.exp(self.gamma * (1 - pt)) * logpt
        elif self.opts.no_EDT:
            loss = -1 * weight * torch.exp(self.gamma * (1 - pt)) * logpt
        else:
            loss = -1 * weight * alphas * torch.exp(self.gamma * (1 - pt)) * logpt
        loss = loss.sum() / (N)


        # if (self.step_counter % self.print_each) == 0:
        #     print(f'Step: {self.step_counter} Loss: {loss.data.cpu().item():.4f}')
        self.step_counter += 1

        return loss



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, weight=None, device=None, opts=None):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        # self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device
        self.weight = weight
        self.opts = opts

        feat_dim = 128
        if not self.opts.deeplab:
            dim_in = 128
        else:
            dim_in = 2048 # 'out'

        # self.conv = nn.Conv2d(in_channels=19, out_channels=1, kernel_size=1),
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))     
        self.projection = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            ).to(self.device)

        self.contrast_mode = 'all' # (ADDED by me); for simclr, 'one', for supcon, 'all'
        

    def forward(self, features, class_labels=None, mask=None):
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        bsz = features.shape[0] // 2
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        features = self.projection(features.to(self.device))
        
        labels = class_labels
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        # print('SUPCON LOSS:', features.shape) # [512, 2, 128]
        # import sys; sys.exit(1)
        
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        # dim 1 concat --> dim 0 concat
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature   # 2x
            anchor_count = contrast_count       # 2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        logits = F.normalize(logits) # for stability (ADDED by me)

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        supcon_loss = loss.view(anchor_count, batch_size).mean()
        return supcon_loss


class FocalLoss2(nn.Module):
    def __init__(self, gamma=0, num_classes=19, ignore_id=19, print_each=20, weight=None, device=None):
        super(FocalLoss2, self).__init__()
        self.num_classes = num_classes
        self.ignore_id = ignore_id
        self.print_each = print_each
        self.step_counter = 0
        self.gamma = gamma
        self.weight = weight
        self.device = device

    def forward(self, input, target, batch, **kwargs):
        if input.shape[-2:] != target.shape[-2:]:
            input = upsample(input, target.shape[-2:])
        target[target == self.ignore_id] = 0  # we can do this because alphas are zero in ignore_id places
        label_distance_weight = batch['label_distance_weight'].to(self.device)
        N = (label_distance_weight.data > 0.).sum()
        if N.le(0):
            return torch.zeros(size=(0,), device=self.device, requires_grad=True).sum()
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        alphas = label_distance_weight.view(-1)
        weight = self.weight[target].view(-1).to(self.device)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.detach().exp()

        loss = -1 * torch.exp(self.gamma * (1 - pt)) * logpt
        loss = loss.sum() / N

        # if (self.step_counter % self.print_each) == 0:
        #     print(f'Step: {self.step_counter} Loss: {loss.data.cpu().item():.4f}')
        self.step_counter += 1

        return loss

from abc import ABC
class PixelContrastLoss(nn.Module, ABC):
    def __init__(self, device=None):
        super(PixelContrastLoss, self).__init__()

        self.device = device
        self.temperature = 0.07
        self.base_temperature = 0.07
        self.ignore_label = 255 # in CE loss
        self.max_samples = 1024
        self.max_views = 2
        self.loss_weight = 1  

        self.contrast_mode = 'all'       

    def _hard_anchor_sampling(self, X, y_hat, y):
        """
        X:      features
        y_hat:  labels
        y:      predictions (channels = # classes, before upsampling resolution)
        """
        print('hard anchor sampling:', X.shape, y_hat.shape, y.shape)
        # hard anchor sampling: torch.Size([bsz, 36864, 128]) torch.Size([bsz, 36864]) torch.Size([bsz, 36864])
        # import sys; sys.exit(1)
        
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).to(self.device)
        y_ = torch.zeros(total_classes, dtype=torch.float).to(self.device)

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii] # labels
            this_y = y[ii] # predict
            this_classes = classes[ii]

            # print(this_y_hat, this_y, this_classes)

            for cls_id in this_classes:
                # print('loop', this_y_hat.shape, this_y.shape, cls_id)
                # print((this_y_hat == cls_id).shape)
                # print((this_y != cls_id).shape)
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero() # different
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero() # same as target

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    print('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):
        # print('inputs to _contrastive:', feats_.shape, labels_.shape)
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().to(self.device)

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        # anchor_feature = contrast_feature # contrast against all (mode='all'); for 'one', use features[:, 0]

        # TODO: modify pixelcontrast loss contrast mode
        # print(feats_.shape)
        # import sys; sys.exit(1)
        if self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count # 2
        elif self.contrast_mode == 'one':
            anchor_feature = feats_[:, 0]
            anchor_count = 1

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits = F.normalize(logits)
        # print(logits, logits.shape)
        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).to(self.device),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        # print('constrastive loss:', loss)
        return loss

    def forward(self, feats, labels=None, predict=None):
        """
        print(feats.shape, labels.shape, predict.shape) 
        torch.Size([bsz, 128, 192, 192]) torch.Size([4, 768, 768]) torch.Size([bsz, #classes, 192, 192])
        """
        _, predict = torch.max(predict, 1) # --> [bsz, 192, 192]
        # feats: [bsz, w, h]
        # labels: [bsz, w, h]

        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]
        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1) # --> torch.Size([bsz, 36864]), 36864 = 192*192
        feats = feats.permute(0, 2, 3, 1)           # [bsz, c, w, h] --> [bsz, w, h, c]
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

        feats_, labels_ = self._hard_anchor_sampling(feats.to(dtype=torch.float32), labels.to(dtype=torch.float32), predict.to(dtype=torch.float32))

        loss = self._contrastive(feats_, labels_)
        return loss



# class SegmentationLosses(object):
#     def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, device=None, n_gpus=1): # ignore_index=255
#         self.ignore_index = ignore_index
#         self.weight = weight
#         self.size_average = size_average
#         self.batch_average = batch_average
#         self.device = device
#         self.n_gpus = n_gpus

#     def build_loss(self, mode='cross_entropy'):
#         """Choices: ['cross_entropy' or 'focal_loss']"""
#         if mode == 'cross_entropy':
#             return self.CrossEntropyLoss
#         elif mode == 'focal_loss':
#             return self.FocalLoss
#         else:
#             raise NotImplementedError

#     def CrossEntropyLoss(self, logit, target):
#         n, c, h, w = logit.size()
#         criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
#                                         size_average=self.size_average)
#         # if self.cuda:
#         #     criterion = criterion.cuda()

#         if self.n_gpus > 1:
#             criterion = nn.Dataparallel(criterion)
#         criterion.to(self.device)

#         loss = criterion(logit, target.long())

#         if self.batch_average:
#             loss /= n

#         return loss

#     def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
#         n, c, h, w = logit.size()
#         criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
#                                         size_average=self.size_average)
#         # if self.cuda:
#         #     criterion = criterion.cuda()

#         if self.n_gpus > 1:
#             criterion = nn.Dataparallel(criterion)
#         criterion.to(self.device)

#         logpt = -criterion(logit, target.long())
#         pt = torch.exp(logpt)
#         if alpha is not None:
#             logpt *= alpha
#         loss = -((1 - pt) ** gamma) * logpt

#         if self.batch_average:
#             loss /= n

#         return loss


# class DisparityLosses(nn.Module):
#     def __init__(self, weight=None, device=None):
#         super(DisparityLosses, self).__init__()
#         self.weight = weight
#         self.device = device
#
#     def forward(self, batch, pyramid_weight, pred_disp_pyramid, gt_disp, mask):
#         disp_loss = 0
#         pyramid_loss = []
#         if 'label_distance_weight' in batch.keys():
#             label_distance_weight = batch['label_distance_weight'].to(self.device)
#             N = (label_distance_weight.data > 0.).sum()
#             if N.le(0):
#                 return torch.zeros(size=(0,), device=self.device, requires_grad=True).sum()
#             alphas = label_distance_weight[mask].view(-1)
#         else:
#             alphas = 1
#
#         for k in range(len(pred_disp_pyramid)):
#             pred_disp = pred_disp_pyramid[k]
#             pyr_weight = pyramid_weight[k]
#
#             if pred_disp.size(-1) != gt_disp.size(-1):
#                 pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
#                 pred_disp = F.interpolate(pred_disp, size=(gt_disp.size(-2), gt_disp.size(-1)),
#                                           mode='bilinear', align_corners=False) * (gt_disp.size(-1) / pred_disp.size(-1))
#                 pred_disp = pred_disp.squeeze(1)  # [B, H, W]
#
#             # curr_loss = F.smooth_l1_loss(pred_disp[mask], gt_disp[mask],
#             #                              reduction='mean')
#
#             t = torch.abs(pred_disp[mask] - gt_disp[mask])
#             curr_loss = alphas * torch.where(t < 1, 0.5 * t ** 2, t - 0.5)
#             curr_loss = curr_loss.sum() / curr_loss.size()[0]
#
#             disp_loss += pyr_weight * curr_loss
#             pyramid_loss.append(curr_loss)
#
#         return disp_loss, pyramid_loss


# def DisparityLosses(pyramid_weight, pred_disp_pyramid, gt_disp, mask):
#     disp_loss = 0

#     for k in range(len(pred_disp_pyramid)):
#         pred_disp = pred_disp_pyramid[k]
#         weight = pyramid_weight[k]

#         if pred_disp.size(-1) != gt_disp.size(-1):
#             pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
#             pred_disp = F.interpolate(pred_disp, size=(gt_disp.size(-2), gt_disp.size(-1)),
#                                       mode='bilinear', align_corners=False) * (gt_disp.size(-1) / pred_disp.size(-1))
#             pred_disp = pred_disp.squeeze(1)  # [B, H, W]

#         curr_loss = F.smooth_l1_loss(pred_disp[mask], gt_disp[mask],
#                                      reduction='mean')
#         disp_loss += weight * curr_loss

#     return disp_loss


# class DispLosses(nn.Module):
#     def __init__(self,weight=None, device=None):
#         super(DispLosses, self).__init__()
#         self.weight = weight
#         self.device = device

#     def forward(self, pred_disp, gt_disp, mask, **kwargs):
#         loss = F.smooth_l1_loss(pred_disp[mask], gt_disp[mask],
#                                      reduction='mean')

#         return loss


# def get_smooth_loss(disp, img):
#     """Computes the smoothness loss for a disparity image
#     The color image is used for edge-aware smoothness
#     """
#     grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
#     grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

#     grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
#     grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

#     grad_disp_x *= torch.exp(-grad_img_x)
#     grad_disp_y *= torch.exp(-grad_img_y)

#     return grad_disp_x.mean() + grad_disp_y.mean()