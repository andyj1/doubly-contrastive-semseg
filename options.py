
from __future__ import absolute_import, division, print_function
from pip._vendor.distlib.compat import raw_input

import os
import argparse
import datetime
import yaml
import shutil

file_dir = os.path.dirname(__file__)  # the directory that options.py resides i


class Options:
	def __init__(self):
		self.parser = argparse.ArgumentParser()

	def _dataset_options(self):
		# Model Options
		self.parser.add_argument("--data_root", type=str, default='/root/dataset',
								 help="path to Dataset ")
		self.parser.add_argument("--dataset", type=str, default='cityscapes',
								 choices=['cityscapes', 'city_lost', 'kitti_2015', 'sceneflow', 'kitti_mix', 'acdc', 'acdc_city'],
								 help='Name of dataset')
		self.parser.add_argument("--num_classes", type=int, default=None,
								 help="num classes (default: auto)")
		self.parser.add_argument("--weather_num", type=int, default=4, help=["4 if acdc', 5 if cityscapes"])
		self.parser.add_argument("--num_workers", type=int, default=0)
		
	def _model_options(self):
		import network
		# deeplab models
		available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
							  not (name.startswith("__") or name.startswith('_')) and callable(
							  network.modeling.__dict__[name])
							  )
		self.parser.add_argument("--model", type=str, default='resnet18',
								 choices=['resnet18',  'mobilenetv2', 'resnet34',
										  'efficientnetb0','enet']+available_models, help='model name')
		self.parser.add_argument("--deeplab", default=False, action='store_true')
		self.parser.add_argument("--separable_conv", action='store_true', default=False,
								 help="apply separable conv to decoder and aspp")
		self.parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

	def _train_options(self):
		# Train Options
		self._train_learning_options()
		self._train_size_options()
		self._train_print_options()
		self._train_resume_options()
		self._validate_options()

	def _train_learning_options(self):
		self.parser.add_argument('--epochs', type=int, default=400, metavar='N',
								 help='number of epochs to train (default: auto)')
		self.parser.add_argument('--start_epoch', type=int, default=0,
								 metavar='N', help='start epochs (default:0)')
		self.parser.add_argument("--total_itrs", type=int, default=30e3,
								 help="epoch number (default: 30k)")

		self.parser.add_argument("--lr", type=float, default=4e-4,
								 help="learning rate (default: 0.001, disparity(0.001...), semantic(0.01...))")
		self.parser.add_argument("--last_lr", type=float, default=1e-6,
								 help="last learning rate, default 1e-6 ")

		self.parser.add_argument("--lr_policy", type=str, default='cos_annealing',
								 choices=['poly', 'step', 'cos', 'cos_step', 'cos_annealing'],
								 help="learning rate scheduler policy")
		self.parser.add_argument("--weight_decay", type=float, default=1e-4,
								 help='weight decay (default: 1e-4)')
		self.parser.add_argument("--optimizer_policy", type=str, default='ADAM', choices=['SGD', 'ADAM'],
								 help="learning rate scheduler policy")

		self.parser.add_argument("--epsilon", type=float, default=1e-1,
								 help='parameter for balancing class weight [1e-2, 2e-2, 5e-2, 1e-1]')
		self.parser.add_argument('--train_semantic', action='store_true', help='train segmentation')

		self.parser.add_argument('--use_balanced_weights', action='store_true', default=True,
								 help='whether to use balanced weights (default: True)')
		self.parser.add_argument("--finetuning", default=False, action='store_true',)

	def _train_size_options(self):
		self.parser.add_argument("--batch_size", type=int, default=8,
								 help='batch size (default: 16)')
		self.parser.add_argument("--val_batch_size", type=int, default=8,
								 help='batch size for validation (default: 4)')
		self.parser.add_argument("--step_size", type=int, default=10000)
		self.parser.add_argument("--crop_size", type=int, default=384)      # width value, in cityscapes dataset height is divided by 2
		self.parser.add_argument("--img_width", type=int, default=1024)
		self.parser.add_argument("--img_height", type=int, default=512)
		self.parser.add_argument("--val_img_width", type=int, default=1920) #2048)
		self.parser.add_argument("--val_img_height", type=int, default=1080) #1024)
		self.parser.add_argument('--base-size', type=int, default=1024,
								 help='base image size')
		self.parser.add_argument("--crop_val", action='store_true', default=False,
								 help='crop validation (default: False)')

	def _train_print_options(self):
		self.parser.add_argument("--gpu_id", type=str, default='0',
								 help="GPU ID")
		self.parser.add_argument("--random_seed", type=int, default=1,
								 help="random seed (default: 1)")
		self.parser.add_argument("--print_freq", type=int, default=10,
								 help="print interval of loss (default: 10)")
		self.parser.add_argument("--summary_freq", type=int, default=40,
								 help="summary interval of loss (default: 100)")
		self.parser.add_argument("--tsne", default=False, action='store_true')
		self.parser.add_argument("--tsne_viz_freq", type=int, default=100,
								 help="summary interval of loss (default: 100)")
		self.parser.add_argument("--val_print_freq", type=int, default=10,
								 help="print interval of validation (default: 10)")
		# self.parser.add_argument("--val_save_freq", type=int, default=30,
		#                          help="print interval of validation (default: 30)")
		self.parser.add_argument("--val_interval", type=int, default=100,
								 help="epoch interval for eval (default: 100)")
		self.parser.add_argument("--download", action='store_true', default=False,
								 help="download datasets")
		self.parser.add_argument("--viz_EDT", action='store_true', default=False,)

		# Log
		self.parser.add_argument('--no_build_summary', action='store_true',
							help='Dont save summary when training to save space')
		self.parser.add_argument('--save_ckpt_freq', default=10, type=int, help='Save checkpoint frequency (epochs)')
		self.parser.add_argument('--wandb', default=None, type=str, help='wandb project name')

	def _train_resume_options(self):
		self.parser.add_argument('--resume', type=str, default=None,
								 help='put the path to resuming file if needed')
		self.parser.add_argument("--continue_training", action='store_true', default=False)
		self.parser.add_argument("--transfer_disparity", action='store_true', default=False)
		self.parser.add_argument('--checkname', type=str, default='test',
								 help='set the checkpoint name')
		self.parser.add_argument('--coarse_features',  action='store_true', default=False)

	def _validate_options(self):
		self.parser.add_argument("--test_only", action='store_true', default=False)
		self.parser.add_argument("--use_test_data", action='store_true', default=False)
		self.parser.add_argument("--weather_condition", default=None, type=str)

	def _stereo_depth_prediction_options(self):

		# Loss
		self.parser.add_argument('--highest_loss_only', action='store_true',
							help='Only use loss on highest scale for finetuning')
		self.parser.add_argument('--with_depth_level_loss', action='store_true', help='train segmentation')

		# md-fusion
		self.parser.add_argument('--not_md_fusion', action='store_true', help='not apply md_fusion')
		self.parser.add_argument('--criterion', type=str, default='none', 
								choices=['supcon_focal', 'supcon_simclr_focal', 
								'plain_focal',
								'pixelcontrast_focal', 'supcon_pixelcontrast_focal', 'supcon_simclr_pixelcontrast_focal',
								'crossentropy','supcon_crossentropy','supcon_simclr_cross_entropy',
								'supcon_none', 'none', 
								'supcon_simclr', 'supcon', 'pixelcontrast_focal'], 
								help='type of criterion')
		self.parser.add_argument('--no_class_weights', action='store_true', help='not apply balancing')
		self.parser.add_argument('--no_EDT', action='store_true', help='not apply balancing')

		# output_dir for test
		self.parser.add_argument('--output_dir', default='output', type=str,
								 help='Directory to save inference results')

		self.parser.add_argument("--new_crop", action='store_true', default=False)
		self.parser.add_argument("--disp_to_obst_ch", action='store_true', default=False)

	def _train_hyper_parameters(self):
		self.parser.add_argument('--amp', action='store_true', default=False)

		self.parser.add_argument("--debug", action='store_true', default=False)
		self.parser.add_argument("--acdc_cityfull", action='store_true', default=False)
		self.parser.add_argument("--use_gamma_correction", action='store_true', default=False)

		self.parser.add_argument("--save_val_results", action='store_true', default=False,
								 help="save segmentation results to \"./results\"")
		self.parser.add_argument("--save_each_results", action='store_true', default=False)
		
		
	def parse(self):
		self._dataset_options()
		self._model_options()
		self._train_options()
		self._stereo_depth_prediction_options()
		self._train_hyper_parameters()

		self.options = self.parser.parse_args()

		# set weather_num=4 if dataset is acdc
		if self.options.dataset == 'acdc' and self.options.weather_num == 5:
			self.options.weather_num = 4

		return self.options
