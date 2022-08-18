import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
import accimage
import torchvision
import torchvision.transforms as transforms
from skimage import color


def pil_loader(path):
	return Image.open(path).convert('RGB')

def accimage_loader(path):
	return accimage.Image(path)
	
def get_img_loader(loader_type):
	if loader_type == 'pil':
		return pil_loader
	elif loader_type == 'accimage':
		torchvision.set_image_backend('accimage')
		return accimage_loader
	else:
		raise ValueError('invalid image loader type: {}'.format(loader_type))

# ---------- for visualization ----------
def rgb_vis(img, mean, std):
	"""
	Args:
		img     : tensor, rgb[-1.0, 1.0], [3, H, W]
	Returns:
		img     : numpy, rgb[0, 255]
	"""
	img = img.data.cpu().numpy()
	for i in range(3):
		img[i, :, :] = img[i, :, :] * std[i] + mean[i]
	img = np.transpose(img, (1, 2, 0)) * 255
	img = np.clip(img, 0, 255)
	img = img.astype(np.uint8)
	return img


def rgbs_vis(imgs, mean, std):
	"""
	Args:
		img     : tensor, rgb[-1.0, 1.0], [B, 3, H, W]
	Returns:
		img     : tensor, rgb[0.0, 1.0]
	"""
	bs = imgs.shape[0]
	imgs_tensor = []
	for i in range(bs):
		img = rgb_vis(imgs[i], mean, std)
		img = Image.fromarray(img)
		img = transforms.ToTensor()(img)
		imgs_tensor.append(img)
	imgs_tensor = torch.stack(imgs_tensor, dim=0)
	return imgs_tensor
