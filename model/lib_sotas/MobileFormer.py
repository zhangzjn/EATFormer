import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.functional import einsum
import numpy as np
from einops import rearrange, repeat

from model import MODEL

def parameter_check(expand_sizes, out_channels, num_token, d_model, in_channel, project_demension, fc_demension, num_classes):
	check_list = [[], []]
	for i in range(len(expand_sizes)):
		check_list[0].extend(expand_sizes[i])
		check_list[1].extend(out_channels[i])
	for i in range(len(check_list[0]) - 1):
		assert check_list[0][i + 1] % check_list[1][
			i] == 0, 'The out_channel should be divisible by expand_size of the next block, due to the expanded DW conv'
	assert num_token > 0, 'num_token should be larger than 0'
	assert d_model > 0, 'd_model should be larger than 0'
	assert in_channel > 0, 'in_channel should be larger than 0'
	assert project_demension > 0, 'project_demension should be larger than 0'
	assert fc_demension > 0, 'fc_demension should be larger than 0'
	assert num_classes > 0, 'num_classes should be larger than 0'


class BottleneckLite(nn.Module):
	'''Proposed in Yunsheng Li, Yinpeng Chen et al., MicroNet, arXiv preprint arXiv: 2108.05894v1'''
	
	def __init__(self, in_channel, expand_size, out_channel, kernel_size=3, stride=1, padding=1):
		super(BottleneckLite, self).__init__()
		self.in_channel = in_channel
		self.expand_size = expand_size
		self.out_channel = out_channel
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.bnecklite = nn.Sequential(
			nn.Conv2d(self.in_channel, self.expand_size, kernel_size=self.kernel_size,
					  stride=self.stride, padding=self.padding, groups=self.in_channel),
			nn.ReLU6(inplace=True),
			nn.Conv2d(self.expand_size, self.out_channel, kernel_size=1, stride=1),
			nn.BatchNorm2d(self.out_channel)
		)
	
	def forward(self, x):
		return self.bnecklite(x)


class MLP(nn.Module):
	'''widths [in_channel, ..., out_channel], with ReLU within'''
	
	def __init__(self, widths, bn=True, p=0.5):
		super(MLP, self).__init__()
		self.widths = widths
		self.bn = bn
		self.p = p
		self.layers = []
		for n in range(len(self.widths) - 2):
			layer_ = nn.Sequential(
				nn.Linear(self.widths[n], self.widths[n + 1]),
				nn.Dropout(p=self.p),
				nn.ReLU6(inplace=True),
			)
			self.layers.append(layer_)
		self.layers.append(
			nn.Sequential(
				nn.Linear(self.widths[-2], self.widths[-1]),
				nn.Dropout(p=self.p)
			)
		)
		self.mlp = nn.Sequential(*self.layers)
		if self.bn:
			self.mlp = nn.Sequential(
				*self.layers,
				nn.BatchNorm1d(self.widths[-1])
			)
	
	def forward(self, x):
		return self.mlp(x)


class DynamicReLU(nn.Module):
	'''channel-width weighted DynamticReLU '''
	'''Yinpeng Chen, Xiyang Dai et al., Dynamtic ReLU, arXiv preprint axXiv: 2003.10027v2'''
	
	def __init__(self, in_channel, control_demension, k=2):
		super(DynamicReLU, self).__init__()
		self.in_channel = in_channel
		self.k = k
		self.control_demension = control_demension
		self.Theta = MLP([control_demension, 4 * control_demension, 2 * k * in_channel], bn=True)
	
	def forward(self, x, control_vector):
		n, _, _, _ = x.shape
		a_default = torch.ones(n, self.k * self.in_channel, device=x.device)
		a_default[:, self.k * self.in_channel // 2:] = torch.zeros(n, self.k * self.in_channel // 2, device=x.device)
		theta = self.Theta(control_vector)
		theta = 2 * torch.sigmoid(theta) - 1
		a = theta[:, 0: self.k * self.in_channel] + a_default
		b = theta[:, self.k * self.in_channel:] * 0.5
		a = rearrange(a, 'n ( k c ) -> n k c', k=self.k)
		b = rearrange(b, 'n ( k c ) -> n k c', k=self.k)
		# x (NCHW), a & b (N, k, C)
		x = einsum('nchw, nkc -> nchwk', x, a) + einsum('nchw, nkc -> nchwk', torch.ones_like(x), b)
		return x.max(4)[0]


class Mobile(nn.Module):
	'''Without shortcut, if stride=2, donwsample, DW conv expand channel, PW conv squeeze channel'''
	
	def __init__(self, in_channel, expand_size, out_channel, token_demension, kernel_size=3, stride=1, k=2):
		super(Mobile, self).__init__()
		self.in_channel, self.expand_size, self.out_channel = in_channel, expand_size, out_channel
		self.token_demension, self.kernel_size, self.stride, self.k = token_demension, kernel_size, stride, k
		
		if stride == 2:
			self.strided_conv = nn.Sequential(
				nn.Conv2d(self.in_channel, self.expand_size, kernel_size=3, stride=2,
						  padding=int(self.kernel_size // 2), groups=self.in_channel),
				nn.BatchNorm2d(self.expand_size),
				nn.ReLU6(inplace=True)
			)
			self.conv1 = nn.Conv2d(self.expand_size, self.in_channel, kernel_size=1, stride=1)
			self.bn1 = nn.BatchNorm2d(self.in_channel)
			self.ac1 = DynamicReLU(self.in_channel, self.token_demension, k=self.k)
			self.conv2 = nn.Conv2d(self.in_channel, self.expand_size, kernel_size=3, stride=1, padding=1,
								   groups=self.in_channel)
			self.bn2 = nn.BatchNorm2d(self.expand_size)
			self.ac2 = DynamicReLU(self.expand_size, self.token_demension, k=self.k)
			self.conv3 = nn.Conv2d(self.expand_size, self.out_channel, kernel_size=1, stride=1)
			self.bn3 = nn.BatchNorm2d(self.out_channel)
		else:
			self.conv1 = nn.Conv2d(self.in_channel, self.expand_size, kernel_size=1, stride=1)
			self.bn1 = nn.BatchNorm2d(self.expand_size)
			self.ac1 = DynamicReLU(self.expand_size, self.token_demension, k=self.k)
			self.conv2 = nn.Conv2d(self.expand_size, self.expand_size, kernel_size=3, stride=1, padding=1,
								   groups=self.expand_size)
			self.bn2 = nn.BatchNorm2d(self.expand_size)
			self.ac2 = DynamicReLU(self.expand_size, self.token_demension, k=self.k)
			self.conv3 = nn.Conv2d(self.expand_size, self.out_channel, kernel_size=1, stride=1)
			self.bn3 = nn.BatchNorm2d(self.out_channel)
	
	def forward(self, x, first_token):
		if self.stride == 2:
			x = self.strided_conv(x)
		x = self.bn1(self.conv1(x))
		x = self.ac1(x, first_token)
		x = self.bn2(self.conv2(x))
		x = self.ac2(x, first_token)
		return self.bn3(self.conv3(x))


class Mobile_Former(nn.Module):
	'''Local feature -> Global feature'''
	
	def __init__(self, d_model, in_channel):
		super(Mobile_Former, self).__init__()
		self.d_model, self.in_channel = d_model, in_channel
		
		self.project_Q = nn.Linear(self.d_model, self.in_channel)
		self.unproject = nn.Linear(self.in_channel, self.d_model)
		self.eps = 1e-10
		self.shortcut = nn.Sequential()
	
	def forward(self, local_feature, x):
		_, c, _, _ = local_feature.shape
		local_feature = rearrange(local_feature, 'n c h w -> n ( h w ) c')  # N, L, C
		project_Q = self.project_Q(x)  # N, M, C
		scores = torch.einsum('nmc , nlc -> nml', project_Q, local_feature) * (c ** -0.5)
		scores_map = F.softmax(scores, dim=-1)  # each m to every l
		fushion = torch.einsum('nml, nlc -> nmc', scores_map, local_feature)
		unproject = self.unproject(fushion)  # N, m, d
		return unproject + self.shortcut(x)


class Former(nn.Module):
	'''Post LayerNorm, no Res according to the paper.'''
	
	def __init__(self, head, d_model, expand_ratio=2):
		super(Former, self).__init__()
		self.d_model = d_model
		self.expand_ratio = expand_ratio
		self.eps = 1e-10
		self.head = head
		assert self.d_model % self.head == 0
		self.d_per_head = self.d_model // self.head
		
		self.QVK = MLP([self.d_model, self.d_model * 3], bn=False)
		self.Q_to_heads = MLP([self.d_model, self.d_model], bn=False)
		self.K_to_heads = MLP([self.d_model, self.d_model], bn=False)
		self.V_to_heads = MLP([self.d_model, self.d_model], bn=False)
		self.heads_to_o = MLP([self.d_model, self.d_model], bn=False)
		self.norm = nn.LayerNorm(self.d_model)
		self.mlp = MLP([self.d_model, self.expand_ratio * self.d_model, self.d_model], bn=False)
		self.mlp_norm = nn.LayerNorm(self.d_model)
	
	def forward(self, x):
		QVK = self.QVK(x)
		Q = QVK[:, :, 0: self.d_model]
		Q = rearrange(self.Q_to_heads(Q), 'n m ( d h ) -> n m d h', h=self.head)  # (n, m, d/head, head)
		K = QVK[:, :, self.d_model: 2 * self.d_model]
		K = rearrange(self.K_to_heads(K), 'n m ( d h ) -> n m d h', h=self.head)  # (n, m, d/head, head)
		V = QVK[:, :, 2 * self.d_model: 3 * self.d_model]
		V = rearrange(self.V_to_heads(V), 'n m ( d h ) -> n m d h', h=self.head)  # (n, m, d/head, head)
		scores = torch.einsum('nqdh, nkdh -> nhqk', Q, K) / (np.sqrt(self.d_per_head) + self.eps)  # (n, h, q, k)
		scores_map = F.softmax(scores, dim=-1)  # (n, h, q, k)
		v_heads = torch.einsum('nkdh, nhqk -> nhqd', V, scores_map)  # (n, h, m, d_p) -> (n, m, h, d_p)
		v_heads = rearrange(v_heads, 'n h q d -> n q ( h d )')
		attout = self.heads_to_o(v_heads)
		attout = self.norm(attout)  # post LN
		attout = self.mlp(attout)
		attout = self.mlp_norm(attout)  # post LN
		return attout  # No res


class Former_Mobile(nn.Module):
	'''Global feature -> Local feature'''
	
	def __init__(self, d_model, in_channel):
		super(Former_Mobile, self).__init__()
		self.d_model, self.in_channel = d_model, in_channel
		
		self.project_KV = MLP([self.d_model, 2 * self.in_channel], bn=False)
		self.shortcut = nn.Sequential()
	
	def forward(self, x, global_feature):
		res = self.shortcut(x)
		n, c, h, w = x.shape
		project_kv = self.project_KV(global_feature)
		K = project_kv[:, :, 0: c]  # (n, m, c)
		V = project_kv[:, :, c:]  # (n, m, c)
		x = rearrange(x, 'n c h w -> n ( h w ) c')  # (n, l, c) , l = h * w
		scores = torch.einsum('nqc, nkc -> nqk', x, K)  # (n, l, m)
		scores_map = F.softmax(scores, dim=-1)  # (n, l, m)
		v_agg = torch.einsum('nqk, nkc -> nqc', scores_map, V)  # (n, l, c)
		feature = rearrange(v_agg, 'n ( h w ) c -> n c h w', h=h)
		return feature + res


class MobileFormerBlock(nn.Module):
	'''main sub-block, input local feature (N, C, H, W) & global feature (N, M, D)'''
	'''output local & global, if stride=2, then it is a downsample Block'''
	
	def __init__(self, in_channel, expand_size, out_channel, d_model, stride=1, k=2, head=8, expand_ratio=2):
		super(MobileFormerBlock, self).__init__()
		
		self.in_channel, self.expand_size, self.out_channel = in_channel, expand_size, out_channel
		self.d_model, self.stride, self.k, self.head, self.expand_ratio = d_model, stride, k, head, expand_ratio
		
		self.mobile = Mobile(self.in_channel, self.expand_size, self.out_channel, self.d_model, kernel_size=3,
							 stride=self.stride, k=self.k)
		self.former = Former(self.head, self.d_model, expand_ratio=self.expand_ratio)
		self.mobile_former = Mobile_Former(self.d_model, self.in_channel)
		self.former_mobile = Former_Mobile(self.d_model, self.out_channel)
	
	def forward(self, local_feature, global_feature):
		z_hidden = self.mobile_former(local_feature, global_feature)
		z_out = self.former(z_hidden)
		x_hidden = self.mobile(local_feature, z_out[:, 0, :])
		x_out = self.former_mobile(x_hidden, z_out)
		return x_out, z_out


class MobileFormer(nn.Module):
	'''Resolution should larger than [2 ** (num_stages + 1) + 7]'''
	'''stem -> bneck-lite -> stages(strided at first block) -> up-project-1x1 -> avg-pool -> fc1 -> scores-fc'''
	
	def __init__(self, expand_sizes=None, out_channels=None,
				 num_token=6, d_model=192, in_channel=3, bneck_exp=32, bneck_out=16,
				 stem_out_channel=16, project_demension=1152, fc_demension=None, num_classes=None):
		super(MobileFormer, self).__init__()
		
		parameter_check(expand_sizes, out_channels, num_token, d_model, in_channel, project_demension, fc_demension, num_classes)
		self.in_channel = in_channel
		self.stem_out_channel = stem_out_channel
		self.num_token, self.d_model = num_token, d_model
		self.num_stages = len(expand_sizes)
		self.bneck_exp = bneck_exp
		self.bneck_out = bneck_out
		self.inter_channel = bneck_out
		self.expand_sizes = expand_sizes
		self.out_channels = out_channels
		self.project_demension, self.fc_demension, self.num_classes = project_demension, fc_demension, num_classes
		
		self.tokens = nn.Parameter(torch.randn(self.num_token, self.d_model), requires_grad=True)
		self.stem = nn.Sequential(
			nn.Conv2d(self.in_channel, self.stem_out_channel, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(self.stem_out_channel),
			nn.ReLU6(inplace=True)
		)
		self.bneck = BottleneckLite(self.stem_out_channel, self.bneck_exp, self.bneck_out, kernel_size=3, stride=1,
									padding=1)
		self.blocks = nn.ModuleList()
		for num_stage in range(self.num_stages):
			num_blocks = len(self.expand_sizes[num_stage])
			for num_block in range(num_blocks):
				if num_block == 0:
					self.blocks.append(
						MobileFormerBlock(self.inter_channel, self.expand_sizes[num_stage][num_block],
										  self.out_channels[num_stage][num_block], self.d_model, stride=2)
					)
					self.inter_channel = self.out_channels[num_stage][num_block]
				else:
					self.blocks.append(
						MobileFormerBlock(self.inter_channel, self.expand_sizes[num_stage][num_block],
										  self.out_channels[num_stage][num_block], self.d_model, stride=1)
					)
					self.inter_channel = self.out_channels[num_stage][num_block]
		
		self.project = nn.Conv2d(self.inter_channel, self.project_demension, kernel_size=1, stride=1)
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.fc = MLP([self.project_demension + self.d_model, self.fc_demension], bn=True)
		self.scores = nn.Linear(self.fc_demension, self.num_classes)
	
	def forward(self, x):
		n, _, _, _ = x.shape
		x = self.stem(x)
		x = self.bneck(x)
		tokens = repeat(self.tokens, 'm d -> n m d', n=n)
		for block in self.blocks:
			x, tokens = block(x, tokens)
		x = self.project(x)
		x = self.avgpool(x).squeeze()
		x = torch.cat([x, tokens[:, 0, :]], dim=-1)
		x = self.fc(x)
		return self.scores(x)


def initNetParams(model):
	for m in model.modules():
		if isinstance(m, nn.Conv2d):
			init.kaiming_normal_(m.weight)
			if m.bias is not None:
				nn.init.constant_(m.bias, 0.1)
		elif isinstance(m, nn.BatchNorm2d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.Linear):
			init.kaiming_normal_(m.weight)
			if m.bias is not None:
				nn.init.constant_(m.bias, 0.1)


@MODEL.register_module
def mobile_former_508(num_classes, pretrained=False, state_dir=None):
	args = {
		'expand_sizes' : [[144, 120], [240, 216], [432, 512, 768, 1056], [1056, 1440, 1440]],
		'out_channels' : [[40, 40], [72, 72], [128, 128, 176, 176], [240, 240, 240]],
		'num_token' : 6, 'd_model' : 192,
		'in_channel' : 3, 'stem_out_channel' : 24,
		'bneck_exp' : 48, 'bneck_out' : 24,
		'project_demension' : 1440, 'fc_demension' : 1920, 'num_classes' : num_classes
	}
	model = MobileFormer(**args)
	if pretrained:
		model.load_state_dict(torch.load(state_dir))
	else:
		initNetParams(model)
	return model


@MODEL.register_module
def mobile_former_294(num_classes, pretrained=False, state_dir=None):
	args = {
		'expand_sizes' : [[96, 96], [144, 192], [288, 384, 576, 768], [768, 1152, 1152]],
		'out_channels' : [[24, 24], [48, 48], [96, 96, 128, 128], [192, 192, 192]],
		'num_token' : 6, 'd_model' : 192,
		'in_channel' : 3, 'stem_out_channel' : 16,
		'bneck_exp' : 32, 'bneck_out' : 16,
		'project_demension' : 1152, 'fc_demension' : 1920, 'num_class' : num_classes
	}
	model = MobileFormer(**args)
	if pretrained:
		model.load_state_dict(torch.load(state_dir))
	else:
		initNetParams(model)
	return model


@MODEL.register_module
def mobile_former_214(num_classes, pretrained=False, state_dir=None):
	args = {
		'expand_sizes' : [[72, 60], [120, 160], [240, 320, 480, 672], [672, 960, 960]],
		'out_channels' : [[20, 20], [40, 40], [80, 80, 112, 112], [160, 160, 160]],
		'num_token' : 6, 'd_model' : 192,
		'in_channel' : 3, 'stem_out_channel' : 12,
		'bneck_exp' : 24, 'bneck_out' : 12,
		'project_demension' : 960, 'fc_demension' : 1600, 'num_classes' : num_classes
	}
	model = MobileFormer(**args)
	if pretrained:
		model.load_state_dict(torch.load(state_dir))
	else:
		initNetParams(model)
	return model


@MODEL.register_module
def mobile_former_151(num_classes, pretrained=False, state_dir=None):
	args = {
		'expand_sizes' : [[72, 48], [96, 96], [192, 256, 384, 528], [528, 768, 768]],
		'out_channels' : [[16, 16], [32, 32], [64, 64, 88, 88], [128, 128, 128]],
		'num_token' : 6, 'd_model' : 192,
		'in_channel' : 3, 'stem_out_channel' : 12,
		'bneck_exp' : 24, 'bneck_out' : 12,
		'project_demension' : 768, 'fc_demension' : 1280, 'num_classes' : num_classes
	}
	model = MobileFormer(**args)
	if pretrained:
		model.load_state_dict(torch.load(state_dir))
	else:
		initNetParams(model)
	return model


@MODEL.register_module
def mobile_former_96(num_classes, pretrained=False, state_dir=None):
	args = {
		'expand_sizes' : [[72], [96, 96], [192, 256, 384], [528, 768]],
		'out_channels' : [[16], [32, 32], [64, 64, 88], [128, 128]],
		'num_token' : 4, 'd_model' : 128,
		'in_channel' : 3, 'stem_out_channel' : 12,
		'bneck_exp' : 24, 'bneck_out' : 12,
		'project_demension' : 768, 'fc_demension' : 1280, 'num_classes' : num_classes
	}
	model = MobileFormer(**args)
	if pretrained:
		model.load_state_dict(torch.load(state_dir))
	else:
		initNetParams(model)
	return model
