import os
import os.path as osp
from PIL import Image
import six
import lmdb
import pickle
import numpy as np

import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS
# from util.data import get_img_loader

from . import DATA


@DATA.register_module
class ImageFolderLMDB(data.Dataset):
	def __init__(self, cfg, train=True, transform=None, target_transform=None):
		self.cfg = cfg
		self.train = train
		self.transform = transform
		self.target_transform = target_transform
		
		self.loader = pickle.loads
		db_path = '{}/{}.lmdb'.format(cfg.data.root, 'train' if train else 'val')
		self.env = lmdb.open(db_path, subdir=osp.isdir(db_path), readonly=True, lock=False, readahead=False, meminit=False)
		self.txn = self.env.begin(write=False)
		self.length = pickle.loads(self.txn.get(b'__len__'))
		self.keys = pickle.loads(self.txn.get(b'__keys__'))

	def __len__(self):
		return self.length
	
	def __getitem__(self, index):
		byteflow = self.txn.get(self.keys[index])
		imgbuf, target = self.loader(byteflow)
		buf = six.BytesIO()
		buf.write(imgbuf)
		buf.seek(0)
		img = Image.open(buf).convert('RGB')
		img = self.transform(img) if self.transform is not None else img
		target = self.target_transform(target) if self.target_transform is not None else target
		return {'img': img, 'target':target}


def folder2lmdb(root, name="train", write_frequency=1000):
	# https://github.com/xunge/pytorch_lmdb_imagenet/blob/master/folder2lmdb.py
	def raw_reader(path):
		with open(path, 'rb') as f:
			bin_data = f.read()
		return bin_data
	
	img_dir = f'{root}/{name}'
	dataset = ImageFolder(root=img_dir, loader=raw_reader)
	data_loader = DataLoader(dataset, num_workers=32, collate_fn=lambda x: x)
	
	lmdb_path = osp.join(root, f'{name}.lmdb')
	db = lmdb.open(lmdb_path, subdir=True, map_size=1099511627776 * 2, readonly=False, meminit=False, map_async=True)
	txn = db.begin(write=True)
	for idx, data in enumerate(data_loader):
		image, label = data[0]
		txn.put(u'{}'.format(idx).encode('ascii'), pickle.dumps((image, label)))
		if (idx + 1) % write_frequency == 0:
			print(f'{name} {idx + 1}/{len(data_loader)}')
			txn.commit()
			txn = db.begin(write=True)
	txn.commit()
	keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
	txn = db.begin(write=True)
	txn.put(b'__keys__', pickle.dumps(keys))
	txn.put(b'__len__', pickle.dumps(len(keys)))
	txn.commit()
	db.sync()
	db.close()


if __name__ == "__main__":
	folder2lmdb('/path/to/imagenet', name='train')
	folder2lmdb('/path/to/imagenet', name='val')
