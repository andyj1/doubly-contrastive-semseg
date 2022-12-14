from torchvision.transforms.functional import normalize
import torch.nn as nn
import numpy as np
import os 

def denormalize(tensor, mean, std):
	mean = np.array(mean)
	std = np.array(std)

	_mean = -mean/std
	_std = 1/std
	return normalize(tensor, _mean, _std)

class Denormalize(object):
	def __init__(self, mean, std):
		mean = np.array(mean)
		std = np.array(std)
		self._mean = -mean/std
		self._std = 1/std

	def __call__(self, tensor):
		if isinstance(tensor, np.ndarray):
			return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
		return normalize(tensor, self._mean, self._std)

def set_bn_momentum(model, momentum=0.1):
	for m in model.modules():
		if isinstance(m, nn.BatchNorm2d):
			m.momentum = momentum

def fix_bn(model):
	for m in model.modules():
		if isinstance(m, nn.BatchNorm2d):
			m.eval()

def mkdir(path):
	if not os.path.exists(path):
		os.mkdir(path)

def read_text_lines(filepath):
	with open(filepath, 'r') as f:
		lines = f.readlines()
	lines = [l.rstrip() for l in lines]
	return lines


def filter_specific_params(kv):
	specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']
	for name in specific_layer_name:
		if name in kv[0]:
			return True
	return False


def filter_semantic_params(kv):
	specific_layer_name = ['segmentation']
	for name in specific_layer_name:
		if name in kv[0]:
			return True
	return False

def filter_feature_extractor_params(kv):
	specific_layer_name = ['feature_extractor']
	for name in specific_layer_name:
		if name in kv[0]:
			return True
	return False


def filter_base_params(kv):
	specific_layer_name = ['offset_conv.weight', 'offset_conv.bias', 'segmentation', 'feature_extractor']
	for name in specific_layer_name:
		if name in kv[0]:
			return False
	return True

def count_parameters(model, opts):
	# import importlib
	# torchsummary_found = importlib.util.find_spec("torchsummary")
	# if torchsummary_found is not None:
	# 	from torchsummary import summary
	# 	print(summary(model.cuda(), (3, opts.crop_size, opts.crop_size), batch_size=8))

	num = sum(p.numel() for p in model.parameters() if p.requires_grad)
	return num


def accuracy(output, target, topk=(1,)):
	import torch
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res


def calculate_meanstd():
    # https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
	import os
	import torch
	from torchvision import datasets, transforms
	from torch.utils.data.dataset import Dataset
	from tqdm.notebook import tqdm
	from time import time
	
	N_CHANNELS = 1
	dataset = datasets.MNIST("./data", download=True, train=True, transform=transforms.ToTensor())
	full_loader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=os.cpu_count())
	
	start_time = time()
	mean = torch.zeros(1)
	std = torch.zeros(1)
	print('==> Computing mean and std...')
	for inputs, _labels in tqdm(full_loader):
		for i in range (N_CHANNELS):
			mean[i] += inputs[:, i, :, :].mean()
			std[i] += inputs[:, i, :, :].std()
	mean.div_(len(dataset))
	mean.div_(len(dataset))
	print(mean, std)
	print('time elapsed:', time()-start_time)