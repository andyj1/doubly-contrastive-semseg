# from genericpath import exists
# from locale import normalize
from metrics.stream_metrics import AverageMeter, TimeAverageMeter
import torch
import torch.nn as nn
import torch.nn.functional as F

# import wandb
import logging
import os
import time
import datetime
import utils
import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import pyplot as plt

from utils.init_trainer import InitOpts
from utils.utils import accuracy
try:
	import wandb
except:
	pass

class Trainer(InitOpts):
	def __init__(self, options):
		super().__init__(options)

		# reduce feature map to linear features
		self.gap = nn.AdaptiveAvgPool2d((1, 1))

		self.backward_time = TimeAverageMeter()
		self.forward_time = TimeAverageMeter()
		self.best_acc = 0

	def train(self):
		logging.info('training...')
		interval_loss,  train_epoch_loss = 0.0, 0.0
		print_cycle, data_cycle = 0.0, 0.0

		# empty the cache
		with torch.cuda.device(self.device):
			torch.cuda.empty_cache()

		# switch to train mode
		self.model.train()
		num_img_tr = len(self.train_loader)

		if self.opts.train_semantic:
			self.criterion.step_counter = 0

		# Learning rate summary
		base_lr = self.optimizer.param_groups[0]['lr']
		self.writer.add_scalar('base_lr', base_lr, self.cur_epochs)
		if self.opts.wandb is not None:
			wandb.log({'base lr': base_lr})
		self.evaluator.reset()

		last_data_time = time.time()
		for i, sample in enumerate(self.train_loader):
			if self.opts.viz_EDT:
				import sys; sys.exit(1)
				
			if 'supcon' in self.opts.criterion:
				sample0, sample1 = sample
				bsz = sample0['left'].shape[0]
				# print(sample0['left'].shape) # [bsz, chan, size, size]
				# import sys; sys.exit(1)
				sample0['left'] = torch.cat([sample0['left'], sample1['left']], dim=0)
				sample = sample0
			
			data_loader_time = time.time() - last_data_time
			data_cycle += data_loader_time
			self.num_iter += 1
			model_start_time = time.time()

			left = sample['left'].to(self.device, dtype=torch.float32)
			labels = sample['label'].to(self.device, dtype=torch.long) if 'label' in sample.keys() else sample['label']
			gt_weather = sample['weather'].to(self.device) if 'weather' in sample.keys() else sample['weather']

			forward_start_time = time.time()
			self.model = self.model.to(left.device)
			self.forward_time.update(time.time() - forward_start_time)

			self.weather_clf = self.weather_clf.to(left.device)

			supcon_flag = bool('supcon' in self.opts.criterion)
			left_seg, left_seg_beforeup, fine_feat, fine_feat0 = self.model(left, return_supcon_feature=supcon_flag)
			# if not self.opts.deeplab:
				# if supcon type, fine_feat is [bsz*2, *], fine_feat0 is [bsz, *]
				# torch.Size([bsz, 19, 768, 768]) torch.Size([bsz, 19, 192, 192]) 
				# torch.Size([bsz, 128, 192, 192]) torch.Size([bsz, 128, 192, 192])
			# else:
				# torch.Size([bsz, 19, 768, 768]) torch.Size([bsz, 19, 192, 192]) 
				# torch.Size([bsz*2, 2048, 48, 48]) torch.Size([bsz, 2048, 192, 192])
			# print(left_seg.shape, left_seg_beforeup.shape, fine_feat.shape, fine_feat0.shape)
			# import sys; sys.exit(1)

			fine_feat = fine_feat.to(left.device) if fine_feat is not None else fine_feat
			fine_feat0 = fine_feat0.to(left.device) if fine_feat0 is not None else fine_feat0
			# classification CE loss
			# if self.opts.criterion=='supcon':
			# 	fine_feat0 = torch.split(fine_feat, [bsz, bsz], dim=0)[0]

			supcon_loss, weather_clf_acc1, seg_loss, pixelcontrast_loss, simclr_loss, ce_loss, loss_weather =\
				 torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.])
			if self.opts.dataset == 'acdc':
				pred_weather = self.weather_clf(fine_feat0)
				loss_weather = F.cross_entropy(pred_weather, gt_weather.view(-1))
				weather_clf_acc1, _ = accuracy(pred_weather, gt_weather, topk=(1,1))
				if self.best_acc < weather_clf_acc1:
						self.best_acc = weather_clf_acc1
			
			if self.opts.criterion ==  'supcon_focal':
				supcon_loss = self.supcon_criterion(fine_feat, 
													class_labels=sample['weather'].to(fine_feat.device), 
													mask=None)
				# semantic segmentation loss
				seg_loss = self.criterion(left_seg, labels, sample)

				total_loss = supcon_loss * 1/self.opts.batch_size + seg_loss * 1.2
				# total_loss = supcon_loss * 1. + seg_loss * 1.
			elif self.opts.criterion ==  'supcon_simclr_focal':
				simclr_loss = self.supcon_criterion(fine_feat, 
													class_labels=None,
													mask=None)
				# semantic segmentation loss
				seg_loss = self.criterion(left_seg, labels, sample)

				total_loss = simclr_loss * 1/self.opts.batch_size + seg_loss * 1.2
				# total_loss = simclr_loss * 1. + seg_loss * 1.
			elif self.opts.criterion == 'pixelcontrast_focal':
				pixelcontrast_loss = self.pixelcontrast_criterion(fine_feat0, 
																	labels=labels, predict=left_seg_beforeup)
				# semantic segmentation loss
				seg_loss = self.criterion(left_seg, labels, sample)
	
				total_loss = pixelcontrast_loss * 1/self.opts.batch_size + seg_loss * 1.2
				# total_loss = pixelcontrast_loss * 1. + seg_loss * 1.
	   
			elif self.opts.criterion == 'supcon_pixelcontrast_focal':
				supcon_loss = self.supcon_criterion(fine_feat, 
													class_labels=sample['weather'].to(fine_feat.device), 
													mask=None)
				# print(fine_feat0.shape, labels.shape, left_seg_beforeup.shape)
				# swiftnet: torch.Size([3, 128, 192, 192]) torch.Size([3, 768, 768]) torch.Size([3, 19, 192, 192])
				# deeplab: torch.Size([3, 2048, 192, 192]) torch.Size([3, 768, 768]) torch.Size([6, 19, 192, 192])
				# import sys; sys.exit(1)
				pixelcontrast_loss = self.pixelcontrast_criterion(fine_feat0, 
																	labels=labels, predict=left_seg_beforeup)
				# print('supcon:', supcon_loss, 'pixelcontrast:', pixelcontrast_loss)
				
				# semantic segmentation loss
				seg_loss = self.criterion(left_seg, labels, sample)

				total_loss = 1/self.opts.batch_size *(supcon_loss + pixelcontrast_loss) + seg_loss * 1.2 			# enet 1 (1overbsz)
				# total_loss = (supcon_loss + pixelcontrast_loss) * 1. + seg_loss * 1.2 							# enet 2
				# total_loss = 1/self.opts.batch_size *(supcon_loss + pixelcontrast_loss) + seg_loss * 1 + (1/self.opts.batch_size)
	
				# total_loss = (supcon_loss + pixelcontrast_loss) * (1/self.opts.batch_size) + seg_loss * 2 		# enet 3
				# total_loss = torch.log(supcon_loss + pixelcontrast_loss) * seg_loss  								# enet #4


			# add simclr pixelcontrast focal if supcon is working
			elif self.opts.criterion == 'supcon_simclr_pixelcontrast_focal':
				simclr_loss = self.supcon_criterion(fine_feat, 
													class_labels=None, 
													mask=None)
				pixelcontrast_loss = self.pixelcontrast_criterion(fine_feat0, 
																	labels=labels, predict=left_seg_beforeup)
				# print('supcon:', supcon_loss, 'pixelcontrast:', pixelcontrast_loss)
				
				# semantic segmentation loss
				seg_loss = self.criterion(left_seg, labels, sample)
				
				# total_loss = simclr_loss * 1. + pixelcontrast_loss * 0.01 + seg_loss * 1.
				total_loss = 1/self.opts.batch_size * (simclr_loss + pixelcontrast_loss) + seg_loss * 1.2 		# checkname: 1 over batchsize

			# elif self.opts.criterion == 'plain_focal':  # already handled in BoundaryAwareFocalLoss
			# 	plain_focal_loss = self.focal_criterion(left_seg, labels, sample)
			# 	total_loss = plain_focal_loss
			elif self.opts.criterion == 'crossentropy':
				ce_loss = self.ce_criterion(left_seg, labels)
				total_loss = ce_loss
			elif self.opts.criterion == 'supcon_crossentropy':
				supcon_loss = self.supcon_criterion(fine_feat, 
													class_labels=sample['weather'].to(fine_feat.device), 
													mask=None)
				ce_loss = self.ce_criterion(left_seg, labels)
				total_loss = ce_loss + supcon_loss
			elif self.opts.criterion == 'supcon_simclr_cross_entropy':
				simclr_loss = self.supcon_criterion(fine_feat, 
													class_labels=None, 
													mask=None)
				ce_loss = self.ce_criterion(left_seg, labels)
				total_loss = ce_loss + supcon_loss
			else:
				# semantic segmentation loss
				seg_loss = self.criterion(left_seg, labels, sample)

				total_loss = seg_loss

			# uncomment to include weather classification CE loss in the total loss
			# total_loss += loss_weather

			interval_loss += total_loss
			train_epoch_loss += total_loss

			backward_start_time = time.time()
			self.optimizer.zero_grad()
			total_loss.backward()
			self.optimizer.step()
			self.backward_time.update(time.time() - backward_start_time)

			one_cycle_time = time.time() - model_start_time
			print_cycle += one_cycle_time

			if self.num_iter % self.opts.print_freq == 0:
				interval_loss = interval_loss / self.opts.print_freq
				logging.info("Epoch: [%3d/%3d][%3d/%3d] DT: %4.2f (s), BT: %4.2f (%4.2f, %4.2f) (s), BT/img: %4.2f (s), loss: %f" %
					  (
						self.cur_epochs, self.opts.epochs, i+1, num_img_tr, 
						data_cycle, 
						print_cycle, self.forward_time.val, self.backward_time.val,
						print_cycle/self.opts.print_freq/self.opts.batch_size, interval_loss)
					)
				self.writer.add_scalar('train/total_loss_print_freq', interval_loss, self.num_iter)
				if self.opts.wandb is not None:
					wandb.log({'train/total_loss_print_freq': interval_loss})
				interval_loss, print_cycle, data_cycle  = 0.0, 0.0, 0.0

			if self.num_iter % self.opts.summary_freq == 0:
				# summary_time_start = time.time()
				self.writer.add_scalar('train/total_loss_summary_freq', total_loss.item(), self.num_iter)
				if self.opts.wandb is not None:
					wandb.log({'train/total_loss_summary_freq': total_loss.item()})
				if self.opts.dataset == 'acdc':
					self.writer.add_scalar('train/weather_loss_summary_freq', loss_weather.item(), self.num_iter)
					self.writer.add_scalar('train/weather_clf_acc_summary_freq', weather_clf_acc1.item(), self.num_iter)

					if self.opts.wandb is not None:
						wandb.log({'train/weather_loss_summary_freq': loss_weather.item()})
						wandb.log({'train/weather_clf_acc_summary_freq': weather_clf_acc1.item()})
				
				# if 'focal' in self.opts.criterion or 'none' in self.opts.criterion:
				if self.opts.criterion != 'crossentropy':
					self.writer.add_scalar('train/sem_loss_summary_freq', seg_loss.item(), self.num_iter)
					if self.opts.wandb is not None:
						wandb.log({'train/sem_loss_summary_freq': seg_loss.item()})
			
				if self.opts.criterion ==  'supcon_focal':
					self.writer.add_scalar('train/supcon_loss_summary_freq', supcon_loss.item(), self.num_iter)
					if self.opts.wandb is not None:
						wandb.log({'train/supcon_loss_summary_freq': supcon_loss.item()})
				elif self.opts.criterion == 'pixelcontrast_focal':
					self.writer.add_scalar('train/pixelcontrast_loss_summary_freq', pixelcontrast_loss.item(), self.num_iter)
					if self.opts.wandb is not None:
						wandb.log({'train/pixelcontrast_loss_summary_freq': pixelcontrast_loss.item()})
				elif self.opts.criterion ==  'plain_focal':
					self.writer.add_scalar('train/plain_focal_loss_summary_freq', seg_loss.item(), self.num_iter)
					if self.opts.wandb is not None:
						wandb.log({'train/plain_focal_loss_summary_freq': seg_loss.item()})
				elif self.opts.criterion == 'supcon_pixelcontrast_focal':
					self.writer.add_scalar('train/supcon_loss_summary_freq', supcon_loss.item(), self.num_iter)
					self.writer.add_scalar('train/pixelcontrast_loss_summary_freq', pixelcontrast_loss.item(), self.num_iter)
					if self.opts.wandb is not None:
						wandb.log({'train/supcon_loss_summary_freq': supcon_loss.item(),
									'train/pixelcontrast_loss_summary_freq': pixelcontrast_loss.item()})
				elif self.opts.criterion ==  'supcon_simclr_focal':
					self.writer.add_scalar('train/simclr_loss_summary_freq', simclr_loss.item(), self.num_iter)
					if self.opts.wandb is not None:
						wandb.log({'train/simclr_loss_summary_freq': simclr_loss.item(),})
				elif self.opts.criterion == 'supcon_crossentropy':
					self.writer.add_scalar('train/supcon_loss_summary_freq', supcon_loss.item(), self.num_iter)
					self.writer.add_scalar('train/ce_loss_summary_freq', ce_loss.item(), self.num_iter)
					if self.opts.wandb is not None:
						wandb.log({'train/supcon_loss_summary_freq': supcon_loss.item(),
									'train/ce_loss_summary_freq': ce_loss.item()})
				elif self.opts.criterion == 'simclr_crossentropy':
					self.writer.add_scalar('train/simclr_loss_summary_freq', simclr_loss.item(), self.num_iter)
					self.writer.add_scalar('train/ce_loss_summary_freq', ce_loss.item(), self.num_iter)
					if self.opts.wandb is not None:
						wandb.log({'train/simclr_loss_summary_freq': simclr_loss.item(),
									'train/ce_loss_summary_freq': ce_loss.item()})
				elif self.opts.criterion == 'crossentropy':
					self.writer.add_scalar('train/ce_loss_summary_freq', ce_loss.item(), self.num_iter)
					if self.opts.wandb is not None:
						wandb.log({'train/ce_loss_summary_freq': ce_loss.item()})

				# summary_time = time.time() - summary_time_start
				# print("summary_time : {}".format(summary_time))
			last_data_time = time.time()
			del total_loss, sample

		train_epoch_loss = train_epoch_loss / num_img_tr
		self.writer.add_scalar('train/total_loss_epoch', train_epoch_loss, self.cur_epochs)
		if self.opts.wandb is not None:
	  		
			wandb.log({'train/total_loss_epoch': train_epoch_loss,})

	def validate(self):
		"""Do validation and return specified samples"""
		logging.info("validation...")
		val_fwd_times = TimeAverageMeter()
		self.evaluator.reset()
		self.time_val = []

		# empty the cache to infer in high res
		with torch.cuda.device(self.device):
			torch.cuda.empty_cache()
		# switch to evaluate mode
		self.model.eval()

		valid_samples, scores, img_id = 0, 0, 0
		num_val = len(self.val_loader)


		gt_weather, labels = torch.tensor([]), torch.tensor([])

		with torch.no_grad():
			start = time.time()
			for i, sample in enumerate(self.val_loader):
				data_time = time.time() - start
				self.time_val_dataloader.append(data_time)
    
				left = sample['left'].to(self.device, dtype=torch.float32)
				self.weather_clf = self.weather_clf.to(left.device)

				if 'label' in sample.keys():
						labels = sample['label']

				if 'weather' in sample.keys():
						gt_weather = sample['weather'].to(self.device)

				valid_samples += 1

				# print('forwarding...')
				# left_seg, pred_weather = self.model(left)
				start_time = time.time()
				left_seg, left_seg_beforeup, fine_feat, fine_feat0 = self.model(left)
				fwt = time.time() - start_time
				
				if self.opts.dataset == 'acdc':
					pred_weather = self.weather_clf(fine_feat)
					self.evaluator.add_batch_weather(gt_weather, pred_weather)

				preds = left_seg.detach().max(dim=1)[1].cpu().numpy()
				# omit if test case
				if not self.opts.use_test_data:
					targets = labels.numpy()
					gt_weather = gt_weather.view(-1)
					self.evaluator.add_batch(targets, preds, gt_weather.cpu().numpy())  # generate confusion matrix

				# add batch time
				val_fwd_times.update(fwt)
				# first batch stucked on some process.. --> time cost is weird on i==0
				if i != 0:
					self.time_val.append(fwt)
	 
					if i % self.opts.val_print_freq == 0:
						# check validation fps
						logging.info("val [%3d/%3d] DT: %.3f(s), BT (bsz=%d): %.3f(s) (BT/img: %.3f(s))" % (
							i, num_val,
							data_time, 
							self.opts.val_batch_size, fwt,
							sum(self.time_val) / len(self.time_val) / self.opts.val_batch_size))

				if self.opts.save_val_results:
					# save all validation results images
					# omit if test case
					# if not self.opts.use_test_data:
					# 	self.save_valid_img_in_results(left, targets, preds, i, None)
					# else:				
					self.save_valid_img_in_results(left, preds, preds, i, sample['frame_name'])
					img_id += 1

				start = time.time()
			del sample

  
		# test validation performance
		score = self.evaluator.get_results()

		save_filename = self.saver.save_file_return()
		weather_acc = self.evaluator.get_weather_results(save_filename)
		# print(weather_acc, type(weather_acc))
		# import sys; sys.exit(1)
		self.performance_test(score, weather_acc, save_filename)

		if not self.opts.test_only:
			self.save_checkpoints_sem(score)

			if self.opts.train_semantic and self.opts.dataset != 'kitti_mix':
				if score['Mean IoU'] > self.best_score:  # save best model
					self.best_score = score['Mean IoU']
					self.best_score_epoch = self.cur_epochs
					self.save_checkpoints_sem(score, is_best=True, best_type='score')
				logging.info('\nbest score {} (epoch: {})'.format(self.best_score, self.best_score_epoch))

		logging.info(f'average fwd time per img: {val_fwd_times.avg:.3f} (s)')

	def test(self):
		self.validate()

	def save_checkpoints_sem(self, score, is_best=False, best_type=None):
		if self.n_gpus > 1:
			model_state = self.model.module.state_dict()
		else:
			model_state = self.model.state_dict()

		self.saver.save_checkpoint({
			'epoch': self.cur_epochs,
			"num_iter": self.num_iter,
			'model_state': model_state,
			'optimizer_state': self.optimizer.state_dict(),
			'score': score,
			'best_score': self.best_score,
			'best_score_epoch': self.best_score_epoch,
		}, is_best, best_type)

	def performance_check_train(self, supcon_weather_loss, total_loss, score):
		if self.opts.train_semantic and self.opts.dataset != 'kitti_mix':
			self.writer.add_scalar('train/mIoU', score["Mean IoU"], self.num_iter)
			self.writer.add_scalar('train/OverallAcc', score["Overall Acc"], self.num_iter)
			self.writer.add_scalar('train/MeanAcc', score["Mean Acc"], self.num_iter)
			self.writer.add_scalar('train/fwIoU', score["FreqW Acc"], self.num_iter)

			if self.opts.wandb is not None:
				wandb.log({'train/mIoU': score["Mean IoU"],
							'train/OverallAcc': score["Overall Acc"],
							'train/MeanAcc': score["Mean Acc"],
							'train/fwIoU': score["FreqW Acc"]})

		if self.opts.supcon:
			self.writer.add_scalar('train/supcon_weather_loss', supcon_weather_loss.item(), self.num_iter)
			self.writer.add_scalar('train/total_loss', total_loss.item(), self.num_iter)

			if self.opts.wandb is not None:
				wandb.log({'train/supcon_weather_loss': supcon_weather_loss.item(),
							'train/total_loss': total_loss.item()})

	def performance_test(self, val_score, weather_acc, save_filename):
		logging.info('Validation:')
		# print('[Epoch: %d]' % (self.cur_epochs))
		if self.opts.train_semantic and self.opts.dataset != 'kitti_mix':
			Acc = self.evaluator.Pixel_Accuracy()
			Acc_class = self.evaluator.Pixel_Accuracy_Class()
			mIoU = self.evaluator.Mean_Intersection_over_Union(save_filename)
			FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
			weather_mIoU = self.evaluator.Mean_Intersection_over_Union_each_weather(save_filename)

			if not self.opts.test_only:
				self.writer.add_scalar('val/mIoU', mIoU, self.cur_epochs)
				self.writer.add_scalar('val/Acc', Acc, self.cur_epochs)
				self.writer.add_scalar('val/Acc_class', Acc_class, self.cur_epochs)
				self.writer.add_scalar('val/fwIoU', FWIoU, self.cur_epochs)
				self.writer.add_scalar('val/Acc_weather', weather_acc, self.cur_epochs)

				if self.opts.wandb is not None:
					wandb.log({'epoch': self.cur_epochs, 
									'val/mIoU': mIoU, 
									'val/Acc': Acc, 
									'val/Acc_class': Acc_class, 
									'val/fwIoU': FWIoU, 
									'val/Acc_weather': weather_acc})

				for key, value in self.val_dst.weather_dict.items():
					self.writer.add_scalar('val/mIoU_' + key, weather_mIoU[str(value)], self.cur_epochs)
					if self.opts.wandb is not None:
						wandb.log({'val/mIoU_' + key: weather_mIoU[str(value)]})

			logging.info(self.evaluator.to_str(val_score))
		else:
			mIoU, Acc, Acc_class, FWIoU = 0, 0, 0, 0

		self.saver.save_val_results_semantic(self.cur_epochs, mIoU, Acc)
		if self.opts.dataset == 'acdc':
			# logging.info("Epoch: [%d/%d] acc: %.6f, class-wise acc: %.6f, mIoU: %.6f, fwIoU: %.6f | weather cls acc: %.6f" 
			# 		% (self.cur_epochs, self.opts.epochs, Acc, Acc_class, mIoU, FWIoU, weather_acc))
			logging.info("Epoch: [%d/%d] weather cls acc: %.4f / 1.0000" % (self.cur_epochs, self.opts.epochs, weather_acc))
			if self.opts.wandb is not None:
				wandb.log({'weather clf acc': weather_acc})
		# else:
		# 	logging.info("Epoch: [%d/%d] acc: %.6f, class-wise acc: %.6f, mIoU: %.6f, fwIoU: %.6f" 
		# 			% (self.cur_epochs, self.opts.epochs, Acc, Acc_class, mIoU, FWIoU))


	def make_directory(self, root, folders):
		if not os.path.exists(os.path.join(root, folders)):
			os.mkdir(os.path.join(root, folders))

	def save_valid_img_in_results(self, left, targets, preds, img_id, sample_frame_name=None):
		file_save_name = sample_frame_name[0].split('.')[0]
		results_top_folder = 'results'
		if self.opts.weather_condition is not None:
			results_top_folder += '_'+self.opts.weather_condition
  
		save_start = time.time()
		if not os.path.exists(os.path.join(self.saver.experiment_dir, results_top_folder)):
			os.mkdir(os.path.join(self.saver.experiment_dir, results_top_folder))

		root_dir = os.path.join(self.saver.experiment_dir, results_top_folder)
		if self.opts.save_each_results:
			self.make_directory(root_dir, 'left_image')
			self.make_directory(root_dir, 'pred_sem')
			self.make_directory(root_dir, 'overlay')
			self.make_directory(root_dir, 'gray_pred_sem')
			if not self.opts.use_test_data:
				self.make_directory(root_dir, 'gt_sem')
		else:
			self.make_directory(root_dir, 'overall')

		# for i in range(len(left)):
		i = 0
		image = left[i].detach().cpu().numpy()
		# image = (self.denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
		image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).transpose(1, 2, 0).astype(np.uint8)
		image_ = image.copy()
		image = Image.fromarray(image)
		# if self.opts.dataset == 'kitti_2015':
		# 	image = image.crop((0, 8, 1242, 8 + 375))

		# omit if test case
		if not self.opts.use_test_data: # gt_sem label map
			target = targets[i]
			target = self.val_loader.dataset.decode_target(target).astype(np.uint8)
			target_ = target.copy()
			target = Image.fromarray(target)
  
		# if self.opts.dataset == 'kitti_2015':
		# 	target = target.crop((0, 8, 1242, 8 + 375))

		pred = preds[i]
		pred = self.val_loader.dataset.decode_target(pred).astype(np.uint8)
		pred_ = pred.copy()
		pred = Image.fromarray(pred)
		# if self.opts.dataset == 'kitti_2015':
		# 	pred = pred.crop((0, 8, 1242, 8 + 375))
		overlay = Image.blend(image, pred, alpha=0.7)

		if self.opts.save_each_results:
			image.save(os.path.join(self.saver.experiment_dir, results_top_folder, 'left_image', file_save_name)+'.png')
			# omit if test case
			if not self.opts.use_test_data:
				target.save(os.path.join(self.saver.experiment_dir, results_top_folder, 'gt_sem', file_save_name)+'.png')
			pred = pred.convert('RGB')
			# print(pred.size) # PIL 1920x1080 (numpy 1080x1920)
			converted_pred = np.zeros(np.array(pred).shape)[:,:,0].transpose() # np 1080x1920 --> np 1920x1080
			w, h = pred.size
			for i in range(w):
				for j in range(h):
					r,g,b = pred.getpixel((i,j))     
					converted_pred[i,j] = self.opts.test_dst.convert_color_to_eval_id((r,g,b))
			converted_pred = Image.fromarray(converted_pred.transpose().astype(np.uint8)) # np 1920x1080 --> np 1080x1920 (PIL 1920x1080)
			from PIL import ImageOps
			converted_pred = ImageOps.grayscale(converted_pred)
			# logging.info(f'====> {len(np.unique(np.array(converted_pred)))}')
			# import sys; sys.exit(1)
	
			import sys
			np.set_printoptions(threshold=sys.maxsize)
			temp = np.array(converted_pred).reshape(1,-1)
			logging.info(f"pred: {np.unique(temp)}, length: {len(np.unique(temp))}")
   
			# if len(np.unique(temp)) > 20: logging.info('error:',len(np.unique(temp)). sample_frame_name[0].split('.')[0])
			# sys.exit(1)

			# if self.opts.use_test_data:
			converted_pred.save(os.path.join(self.saver.experiment_dir, results_top_folder, 'gray_pred_sem', file_save_name)+'.png')
			pred.save(os.path.join(self.saver.experiment_dir, results_top_folder, 'pred_sem', file_save_name)+'.png')
			overlay.save(os.path.join(self.saver.experiment_dir, results_top_folder, 'overlay', file_save_name)+'.png')
			# else:
			# 	pred.save(os.path.join(self.saver.experiment_dir, results_top_folder, 'pred_sem', '%d_pred_sem.png' % img_id))
			# 	overlay.save(os.path.join(self.saver.experiment_dir,results_top_folder, 'overlay', '%d_overlay.png' % img_id))
			
   			# another reference: https://github.com/DeepSceneSeg/EfficientPS/blob/master/tools/cityscapes_save_predictions.py
			# out = Image.blend(img, sem_img, 0.5).convert(mode="RGBA")
            # out = Image.alpha_composite(out, contours_img)
            # out.convert(mode="RGB").save(out_path)
		else:
			# omit target if test case
			if not self.opts.use_test_data:
				store_img = np.concatenate([i.astype(np.uint8) for i in [image_, target_, pred_]], axis=0)
			else:
				store_img = np.concatenate([i.astype(np.uint8) for i in [pred_]], axis=0)
				# store_img = np.concatenate([i.astype(np.uint8) for i in [image_, pred_]], axis=0)
			store_img = Image.fromarray(store_img)
			store_img.thumbnail((720, 720))
			store_img.save(os.path.join(self.saver.experiment_dir, results_top_folder, 'overall', '%d_overall.png' % img_id))


		save_end = time.time()
		logging.info("=> saved to {}.png | time: {:.3f} (s)".format(file_save_name, (save_end - save_start)))


	# def save_valid_img_in_results_kitti(self, left, right, targets, preds, pred_disp, gt_disp, img_id, pseudo_disp=None):
	# 	if not os.path.exists(os.path.join(self.saver.experiment_dir, 'results')):
	# 		os.mkdir(os.path.join(self.saver.experiment_dir, 'results'))

	# 	root_dir = os.path.join(self.saver.experiment_dir, 'results')
	# 	self.make_directory(root_dir, 'left_image')
	# 	self.make_directory(root_dir, 'right_image')
	# 	self.make_directory(root_dir, 'gt_sem')
	# 	self.make_directory(root_dir, 'pred_sem')
	# 	self.make_directory(root_dir, 'overlay')

	# 	# draw first images in each batch to save memory
	# 	i = 0
	# 	image = left[i].detach().cpu().numpy()
	# 	right_image = right[i].detach().cpu().numpy()
	# 	# image = (self.denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
	# 	image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).transpose(1, 2, 0).astype(np.uint8)
	# 	image = Image.fromarray(image)
	# 	if self.opts.dataset == 'kitti_2015':
	# 		image = image.crop((0, 8, 1242, 8 + 375))
	# 	image.save(os.path.join(self.saver.experiment_dir, 'results', 'left_image', '%d_left_image.png' % img_id))

	# 	right_image = (
	# 				(right_image - np.min(right_image)) / (np.max(right_image) - np.min(right_image)) * 255).transpose(
	# 		1, 2, 0).astype(
	# 		np.uint8)
	# 	right_image = Image.fromarray(right_image)
	# 	if self.opts.dataset == 'kitti_2015':
	# 		right_image = right_image.crop((0, 8, 1242, 8 + 375))
	# 	right_image.save(
	# 		os.path.join(self.saver.experiment_dir, 'results', 'right_image', '%d_right_image.png' % img_id))

	# 	if self.opts.train_semantic:
	# 		target = targets[i]
	# 		target = self.val_loader.dataset.decode_target(target).astype(np.uint8)
	# 		target = Image.fromarray(target)
	# 		if self.opts.dataset == 'kitti_2015':
	# 			target = target.crop((0, 8, 1242, 8 + 375))
	# 		target.save(
	# 			os.path.join(self.saver.experiment_dir, 'results', 'gt_sem', '%d_gt_sem.png' % img_id))

	# 		pred = preds[i]
	# 		pred = self.val_loader.dataset.decode_target(pred).astype(np.uint8)
	# 		pred = Image.fromarray(pred)
	# 		if self.opts.dataset == 'kitti_2015':
	# 			pred = pred.crop((0, 8, 1242, 8 + 375))
	# 		pred.save(os.path.join(self.saver.experiment_dir, 'results', 'pred_sem', '%d_pred_sem.png' % img_id))

	# 		overlay = Image.blend(image, pred, alpha=0.7)
	# 		overlay.save(os.path.join(self.saver.experiment_dir, 'results', 'overlay', '%d_overlay.png' % img_id))

	def calculate_estimate(self, epoch, iter):
		num_img_tr = len(self.train_loader)
		num_img_val = len(self.val_loader)
		estimate = int((self.data_time_t.avg + self.batch_time_t.avg) * \
					   (num_img_tr * self.opts.epochs - (
							   iter + 1 + epoch * num_img_tr))) + \
				   int(self.batch_time_e.avg * num_img_val * (
						   self.opts.epochs - (epoch)))
		return str(datetime.timedelta(seconds=estimate))

	def resize_pred_disp(self, pred_disp, gt_disp):
		if pred_disp.size(-1) < gt_disp.size(-1):
			pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
			pred_disp = F.interpolate(pred_disp, (gt_disp.size(-2), gt_disp.size(-1)),
									  mode='bilinear', align_corners=False) * (
								gt_disp.size(-1) / pred_disp.size(-1))
			pred_disp = pred_disp.squeeze(1)  # [B, H, W]
		return pred_disp