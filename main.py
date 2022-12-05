from __future__ import absolute_import, division, print_function

import time
import warnings

import utils
from options import Options
from trainer import *
from utils import logger

warnings.filterwarnings("ignore")

def init_wandb_setting(opts):
	import wandb
	wandb.tensorboard.patch(root_logdir=opts.experiment_dir)
	wandb.init(project=opts.wandb, sync_tensorboard=True, pytorch=True)
	
if __name__ == '__main__':
	options = Options()
	opts = options.parse()


	utils.logger.seed_all_rng(seed=opts.random_seed)

	trainer = Trainer(opts)

	if opts.wandb is not None:
		init_wandb_setting(opts)
	
	TIME_STAMP = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
	
	if torch.cuda.is_available(): torch.backends.cudnn.benchmark = True
	logger.setup_logger("weathernet", opts.experiment_dir, filename=f'{opts.model}_{opts.dataset}_{TIME_STAMP}_log.txt')

	if opts.tsne:
		from utils import tsne
		opts.criterion = 'none' # need not have 'supcon' in criterion (supcon --> creates two feature sets)
		tsne.Viz(opts)
		import sys; sys.exit(1)

	if opts.test_only:
		if opts.resume is None:
			raise RuntimeError("=> no checkpoint found...")
		else:
			print("checkpoint found at '{}'" .format(opts.resume))
		trainer.validate()
		# trainer.test()
	else:
		for epoch in range(trainer.opts.start_epoch, trainer.opts.epochs):
			epoch_start_time = time.time()
			trainer.train()
			trainer.validate()
			trainer.scheduler.step()
			trainer.cur_epochs += 1
			epoch_end_time = time.time()
			print("time for epoch (train+val): {} (s)\n".format(epoch_end_time - epoch_start_time))

		print('=> Finished training.\n\n')
		trainer.writer.close()
	# wandb.finish()