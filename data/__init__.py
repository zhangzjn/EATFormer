import glob
import importlib
import torch
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torchvision import transforms

from util.registry import Registry
from data.samplers import RASampler
TRANSFORMS = Registry('Transforms')
DATA = Registry('Data')

files = glob.glob('data/[!_]*.py')
for file in files:
	model_lib = importlib.import_module(file.split('.')[0].replace('/', '.'))


def get_transforms(cfg, train, cfg_transforms):
	transform_list = []
	for t in cfg_transforms:
		t = {k: v for k, v in t.items()}
		t_type = t.pop('type')
		t_tran = TRANSFORMS.get_module(t_type)(**t)
		transform_list.extend(t_tran) if isinstance(t_tran, list) else transform_list.append(t_tran)
	transform_out = TRANSFORMS.get_module('Compose')(transform_list)
	
	if train:
		if cfg.size <= 32 and cfg.type == 'CLS':
			transform_out[0] = transforms.RandomCrop(cfg.size, padding=4)
	return transform_out


def get_dataset(cfg):
	train_transforms = get_transforms(cfg, train=True, cfg_transforms=cfg.data.train_transforms)
	test_transforms = get_transforms(cfg, train=False, cfg_transforms=cfg.data.test_transforms)
	train_set = DATA.get_module(cfg.data.name)(cfg, train=True, transform=train_transforms)
	test_set = DATA.get_module(cfg.data.name)(cfg, train=False, transform=test_transforms)
	return train_set, test_set


def get_loader(cfg):
	train_set, test_set = get_dataset(cfg)
	if cfg.dist:
		if cfg.data.sampler == 'naive':
			sampler = DistributedSampler
		elif cfg.data.sampler == 'ra':
			sampler = RASampler
		else:
			raise NotImplementedError("sampler '{}' is not implemented".format(cfg.data.sampler))
		train_sampler = sampler(train_set, shuffle=True)
		test_sampler = sampler(test_set, shuffle=False)
		
	else:
		train_sampler = None
		test_sampler = None

	train_loader = torch.utils.data.DataLoader(dataset=train_set,
											   batch_size=cfg.trainer.data.batch_size_per_gpu,
											   shuffle=(train_sampler is None),
											   sampler=train_sampler,
											   num_workers=cfg.trainer.data.num_workers_per_gpu,
											   pin_memory=cfg.trainer.data.pin_memory,
											   drop_last=cfg.trainer.data.drop_last,
											   persistent_workers=cfg.trainer.data.persistent_workers)
	test_loader = torch.utils.data.DataLoader(dataset=test_set,
											  batch_size=cfg.trainer.data.batch_size_per_gpu_test,
											  shuffle=False,
											  sampler=test_sampler,
											  num_workers=cfg.trainer.data.num_workers_per_gpu,
											  pin_memory=cfg.trainer.data.pin_memory,
											  drop_last=False,
											  persistent_workers=cfg.trainer.data.persistent_workers)
	return train_loader, test_loader
