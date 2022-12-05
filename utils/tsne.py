import random
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.manifold import TSNE
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.init_trainer import InitOpts

# CLASS_SPECIFIC = True
CLASS_SPECIFIC = False
MARKER_SIZE = 0.1
# ---------------------------------------
# # append features for TSNE visualization
# if self.opts.tsne: # and (self.num_iter % self.opts.tsne_viz_freq == 0):
# 	# self.tsne_features = torch.from_numpy(self.tsne_features)
# 	current_features = F.adaptive_avg_pool2d(fine_feat0, output_size=(1,1))
# 	current_features = torch.flatten(current_features, 1)

# 	if self.tsne_features is not None:
# 		self.tsne_features = np.concatenate((self.tsne_features, current_features.detach().cpu().numpy()))
# 	else:
# 		self.tsne_features = current_features.detach().cpu().numpy()
	
# 	print('cumulative tsne features:', self.tsne_features.shape)
# 	if self.tsne_features.shape[0] == 4*self.opts.batch_size: # batch size of 4 * 64 = 256
# 		save_filename_path = os.path.join(self.saver.experiment_dir, 'tsne.png')
# 		print('visualizing tsne...')
# 		visualize_tsne(self.tsne_features, sample['weather'], save_path=save_filename_path)
# 		print('DONE visualizing tsne.')
# 		import sys; sys.exit(1)
# 	else:
# 		self.model.to('cpu')
# 		del sample, current_features, left_seg, left_seg_beforeup, fine_feat, fine_feat0, labels, gt_weather
# 		continue
# 	continue
# ---------------------------------------
   
class Viz(InitOpts):
	def __init__(self, opts):
		self.opts = opts
		# self.opts.data_root = '/root/dataset/'
		self.opts.data_root = '/mnt/f'
		super().__init__(opts)

	
		# random seeds
		seed = 10
		random.seed(seed)
		torch.manual_seed(seed)
		np.random.seed(seed)
		
		features, labels = self.get_features()
		tsne = TSNE(n_components=2).fit_transform(features)
		save_path = os.path.join(self.saver.experiment_dir, 'tsne.png')
		self.visualize_tsne(tsne, labels, save_path)

	def get_features(self):
		self.model = self.model.to(self.device)
		self.model.eval()
  
		features = None
		labels = []
		
		for sample in tqdm(self.train_loader, desc='model inference'):
			images = sample['left'].to(self.device)
			# print(type(sample['weather']), sample['weather'])
			
			with torch.no_grad():
				_, _, _, current_features = self.model(images, False)
			# print(current_features.shape) # [bsz, feat_dim, h, w]

			# === obtain class-specific 128-dim features ====
			if CLASS_SPECIFIC:
				labels += [item[0] for item in sample['weather'].numpy().tolist()] # [bsz]
				# print(len(labels))
				# import sys; sys.exit(1)

				# global average pool over the image size (h,w)
				current_features = torch.nn.functional.adaptive_avg_pool2d(current_features, output_size=(1,1))
				current_features = torch.flatten(current_features, 1)
				current_features = current_features.detach().cpu().numpy() # [bsz, feat_dim] 

				# print(type(current_features))
				
				if features is not None:
					features = np.concatenate((features, current_features))
				else:
					features = current_features
			else:
				bsz, _, h, w = current_features.shape
				labels = sample['label'] # [bsz, 768, 768]
				
				labels = labels.unsqueeze(1).float().clone() # [bsz, 1, 768, 768]
				labels = torch.nn.functional.interpolate(labels, size=(h,w), mode='nearest') # [bsz, 1, 192, 192]

				# === obtain pixel-specific 128-dim features ====
				pixel_labels = []
				current_pixel_features = []
				for b in range(bsz):
					for j in range(h):
						for i in range(w):
							current_labels = labels[b, 0, j, i].detach().cpu().numpy()
							if current_labels == 255:
								continue
							else:
								pixel_labels.append(np.expand_dims(current_labels, axis=0))

								each_pixel_features = torch.flatten(current_features[b,:,j,i]).detach().cpu().numpy() # [1, feat_dim]
								current_pixel_features.append(np.expand_dims(each_pixel_features, axis=0))
				
				# PIXELS = 30000
				labels = pixel_labels #[:PIXELS]
				current_pixel_features = current_pixel_features #[:PIXELS]
				features =  torch.flatten(torch.from_numpy(np.array(current_pixel_features)),1).detach().cpu().numpy()
				break
				# print('TOTAL pixel label:', len(pixel_labels), 'pixel feature:', torch.flatten(torch.from_numpy(np.array(current_pixel_features)),1).shape)
			# import sys; sys.exit(1)
			
			
			
			# if features.shape[0] > 1024:
			# 	break

			del sample

		return features, labels
	
	def scale_to_01_range(self, x):
		# compute the distribution range
		value_range = (np.max(x) - np.min(x))

		# move the distribution so that it starts from zero by extracting the minimal value from all its values
		starts_from_zero = x - np.min(x)

		# make the distribution fit [0; 1] by dividing by its range
		return starts_from_zero / value_range
	
	def visualize_tsne(self, features, labels, save_path='/root/dataset/tsne.png'):
		# features: torch.tensor
		# class-specific
		# colors_per_class = {    
		# 	0 : [254, 202, 87],  # 'fog'
		# 	1 : [255, 107, 107], # 'night'
		# 	2 : [10, 189, 227],  # 'rain'
		# 	3 : [255, 159, 243], # 'snow'
		# 	# 4 : [0, 0, 0]      # 'sunny'   
		# }

		# pixel-specific
		colors_per_class = {
			0: [128, 64, 128],
			1: [244, 35, 232],
			2: [70, 70, 70],
			3: [102, 102, 156],
			4: [190, 153, 153],
			5: [153, 153, 153],
			6: [250, 170, 30],
			7: [220, 220, 0],
			8: [107, 142, 35],
			9: [152, 251, 152],
			10: [70, 130, 180],
			11:	[220, 20, 60],
			12: [255, 0, 0],
			13: [0, 0, 142],
			14: [0, 0, 70],
			15: [0, 60, 100],
			16: [0, 80, 100],
			17: [0, 0, 230],
			18: [119, 11, 32],
			255: [0, 0, 0]
		}

		weather_dict = self.evaluator.weather_dict # str key  type
  
		# change key type from str to int (need to change later)
		label_dict = dict()
		if CLASS_SPECIFIC:
			for key, item in weather_dict.items():
				label_dict[int(key)] = item
			# weather_dict = {
			# 		0: 'fog',
			# 		1: 'night',
			# 		2: 'rain',
			# 		3: 'snow',
			# 		4: 'sunny'
			# 	}
		else:
			label_dict = {c.train_id: c.name for c in self.opts.train_dst.classes}
		# print(label_dict)
		# import sys; sys.exit(1)
  
		from sklearn.manifold import TSNE  # cpu only
		tsne = TSNE(n_components=2).fit_transform(features)
		
		# extract x and y coordinates representing the positions of the images on T-SNE plot
		tx = tsne[:, 0]
		ty = tsne[:, 1]

		# scale and move the coordinates so they fit [0; 1] range
		tx = self.scale_to_01_range(tx)
		ty = self.scale_to_01_range(ty)

		# visualize the plot: samples as colored points
		# print('tsne:', tx.shape, ty.shape) # [bsz, 1], [bsz, 1]
		
		# initialize matplotlib plot
		fig = plt.figure()
		ax = fig.add_subplot(111)
		# print(labels, type(labels))
		# if len(labels.shape) > 1:
		# 	labels = labels.view(-1, 1).squeeze(-1)
			
		# for every class, we'll add a scatter plot separately
		for label in colors_per_class:
			# find the samples of the current class in the data
			
			indices = [i for i, l in enumerate(labels) if l == label]

			# extract the coordinates of the points of this class only
			# print(tx.shape, ty.shape, len(indices))
			
			current_tx = np.take(tx, indices)
			current_ty = np.take(ty, indices)

			# convert the class color to matplotlib format:
			# BGR -> RGB, divide by 255, convert to np.array
			color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255

			# add a scatter plot with the correponding color and label
			ax.scatter(current_tx, current_ty, c=color, s=MARKER_SIZE)#, label=label_dict[label])

		# build a legend using the labels we set previously
		ax.legend(loc='best')

		# finally, show the plot
		plt.savefig(save_path, dpi=300)
		print('saved to:', save_path)
		# plt.show()
