import torch
import torch.nn as nn
import torch.nn.functional as F
from . import LOSS
from model import get_model

__all__ = ['CE', 'LabelSmoothingCE', 'SoftTargetCE', 'CLSKDLoss']


@LOSS.register_module
class CE(nn.CrossEntropyLoss):
	def __init__(self, lam=1):
		super(CE, self).__init__()
		self.lam = lam

	def forward(self, input, target):
		return super(CE, self).forward(input, target) * self.lam


@LOSS.register_module
class LabelSmoothingCE(nn.Module):
	"""
	NLL loss with label smoothing.
	"""
	def __init__(self, smoothing=0.1, lam=1):
		"""
		Constructor for the LabelSmoothing module.
		:param smoothing: label smoothing factor
		"""
		super(LabelSmoothingCE, self).__init__()
		assert smoothing < 1.0
		self.smoothing = smoothing
		self.lam = lam
		self.confidence = 1. - smoothing

	def forward(self, x, target):
		logprobs = F.log_softmax(x, dim=-1)
		nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
		nll_loss = nll_loss.squeeze(1)
		smooth_loss = -logprobs.mean(dim=-1)
		loss = self.confidence * nll_loss + self.smoothing * smooth_loss
		return loss.mean() * self.lam


@LOSS.register_module
class SoftTargetCE(nn.Module):
	def __init__(self, lam=1, fp32=False):
		super(SoftTargetCE, self).__init__()
		self.lam = lam
		self.fp32 = fp32

	def forward(self, x, target):
		if self.fp32:
			loss = torch.sum(-target * F.log_softmax(x.float(), dim=-1), dim=-1)
		else:
			loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
		return loss.mean() * self.lam


@LOSS.register_module
class CLSKDLoss(torch.nn.Module):
	def __init__(self, cfg, kd_type, tau=1.0, lam=1):
		super().__init__()
		self.teacher_model = get_model(cfg)
		self.teacher_model.cuda()
		self.teacher_model.eval()
		assert kd_type in ['soft', 'hard']
		self.kd_type = kd_type
		self.tau = tau
		self.lam = lam

	def forward(self, inputs, outputs_kd):
		with torch.no_grad():
			teacher_outputs = self.teacher_model(inputs)
		if self.kd_type == 'soft':
			# taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
			# with slight modifications
			distillation_loss = F.kl_div(F.log_softmax(outputs_kd / self.tau, dim=1),
										 F.log_softmax(teacher_outputs / self.tau, dim=1),
										 reduction='sum', log_target=True) * (self.tau * self.tau) / outputs_kd.numel()
		elif self.kd_type == 'hard':
			distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))
		else:
			raise ValueError(f'invalid distillation type: {self.kd_type}')
		return distillation_loss * self.lam
