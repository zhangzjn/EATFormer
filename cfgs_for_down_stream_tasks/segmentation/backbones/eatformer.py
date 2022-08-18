import math
from functools import partial

from einops import rearrange, reduce, repeat
from torchvision.ops import DeformConv2d
from timm.models.layers.activations import *
from timm.models.layers import DropPath, trunc_normal_
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES

init_alpha_value = 1e-3
init_scale_values = 1e-4


# ========== For Common ==========
class LayerNormConv(nn.Module):

	def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
		super().__init__()
		self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

	def forward(self, x):
		x = rearrange(x, 'b c h w -> b h w c').contiguous()
		x = self.norm(x)
		x = rearrange(x, 'b h w c -> b c h w').contiguous()
		return x


def get_norm(norm_layer='in_1d'):
	eps = 1e-6
	norm_dict = {
		'none': nn.Identity,
		'in_1d': partial(nn.InstanceNorm1d, eps=eps),
		'in_2d': partial(nn.InstanceNorm2d, eps=eps),
		'in_3d': partial(nn.InstanceNorm3d, eps=eps),
		'bn_1d': partial(nn.BatchNorm1d, eps=eps),
		'bn_2d': partial(nn.BatchNorm2d, eps=eps),
		# 'bn_2d': partial(nn.SyncBatchNorm, eps=eps),
		'bn_3d': partial(nn.BatchNorm3d, eps=eps),
		'gn': partial(nn.GroupNorm, eps=eps),
		'ln': partial(nn.LayerNorm, eps=eps),
		'lnc': partial(LayerNormConv, eps=eps),
	}
	return norm_dict[norm_layer]


def get_act(act_layer='relu'):
	act_dict = {
		'none': nn.Identity,
		'sigmoid': Sigmoid,
		'swish': Swish,
		'mish': Mish,
		'hsigmoid': HardSigmoid,
		'hswish': HardSwish,
		'hmish': HardMish,
		'tanh': Tanh,
		'relu': nn.ReLU,
		'relu6': nn.ReLU6,
		'prelu': PReLU,
		'gelu': GELU,
		'silu': nn.SiLU
	}
	return act_dict[act_layer]


# ========== Individual ==========
class MLP(nn.Module):

	def __init__(self, in_dim, hid_dim=None, out_dim=None, act_layer='gelu', drop=0.):
		super().__init__()
		out_dim = out_dim or in_dim
		hid_dim = hid_dim or in_dim
		self.fc1 = nn.Conv2d(in_dim, hid_dim, kernel_size=1, stride=1, padding=0)
		self.act = get_act(act_layer)()
		self.fc2 = nn.Conv2d(hid_dim, out_dim, kernel_size=1, stride=1, padding=0)
		self.drop = nn.Dropout(drop)

	def forward(self, x):
		x = self.fc1(x)
		x = self.act(x)
		x = self.drop(x)
		x = self.fc2(x)
		x = self.drop(x)
		return x


class FFN(nn.Module):

	def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer='gelu', norm_layer='lnc'):
		super().__init__()
		self.norm = get_norm(norm_layer)(dim)
		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
		hid_dim = int(dim * mlp_ratio)
		self.mlp = MLP(in_dim=dim, hid_dim=hid_dim, out_dim=dim, act_layer=act_layer, drop=drop)
		self.gamma_mlp = nn.Parameter(init_scale_values * torch.ones((dim)), requires_grad=True)

	def forward(self, x):
		shortcut = x
		x = self.norm(x)
		x = shortcut + self.drop_path(self.gamma_mlp.unsqueeze(0).unsqueeze(2).unsqueeze(3) * self.mlp(x))
		return x


# ========== Global and Local Populations ==========
class MSA(nn.Module):
	def __init__(self, dim, dim_head, qkv_bias=False, attn_drop=0., proj_drop=0.):
		super().__init__()
		self.dim_head = dim_head
		self.num_head = dim // dim_head
		self.scale = self.dim_head ** -0.5

		self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, stride=1, padding=0, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, x):
		B, C, H, W = x.shape

		qkv = self.qkv(x)
		qkv = rearrange(qkv, 'b (qkv heads dim_head) h w -> qkv b heads (h w) dim_head', qkv=3, heads=self.num_head,
						dim_head=self.dim_head).contiguous()
		q, k, v = qkv[0], qkv[1], qkv[2]

		attn = (q @ k.transpose(-2, -1)) * self.scale
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)

		x = attn @ v
		x = rearrange(x, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head,
					  dim_head=self.dim_head, h=H, w=W).contiguous()
		x = self.proj(x)
		x = self.proj_drop(x)
		return x


class MSA_OP(nn.Module):

	def __init__(self, dim, dim_head, window_size, qkv_bias=False, attn_drop=0., proj_drop=0., init_scale_values=1e-4):
		super().__init__()
		assert dim % dim_head == 0
		self.window_size = window_size if isinstance(window_size, list) else [window_size, window_size]
		self.msa = MSA(dim, dim_head, qkv_bias, attn_drop, proj_drop)
		self.gamma_msa = nn.Parameter(init_scale_values * torch.ones((dim)), requires_grad=True)

	def forward(self, x):
		B, C, H, W = x.shape
		if self.window_size[0] <= 0:
			window_size_W, window_size_H = W, H
		else:
			window_size_W, window_size_H = self.window_size[1], self.window_size[0]
		pad_l, pad_t = 0, 0
		pad_r = (window_size_W - W % window_size_W) % window_size_W
		pad_b = (window_size_H - H % window_size_H) % window_size_H
		x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))

		n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
		x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous()
		x = self.gamma_msa.unsqueeze(0).unsqueeze(2).unsqueeze(3) * self.msa(x)
		x = rearrange(x, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()

		if pad_r > 0 or pad_b > 0:
			x = x[:, :, :H, :W].contiguous()

		return x


class DMSA(nn.Module):
	def __init__(self, dim, dim_head, kernel_size, stride, qkv_bias=False, attn_drop=0., proj_drop=0., d_groups=3):
		super().__init__()
		assert dim % dim_head == 0
		self.kernel_size = kernel_size
		self.stride = stride
		self.dim = dim
		self.dim_head = dim_head
		self.num_head = dim // dim_head
		self.scale = self.dim_head ** -0.5
		self.d_groups = d_groups
		self.n_group_dim = self.dim // self.d_groups
		self.offset_range_factor = 2

		self.conv_offset_modulation = nn.Sequential(
			nn.Conv2d(self.n_group_dim, self.n_group_dim, self.kernel_size, self.stride, self.kernel_size // 2,
					  groups=self.n_group_dim),
			get_norm('bn_2d')(self.n_group_dim),
			nn.GELU(),
			nn.Conv2d(self.n_group_dim, 3, 1, 1, 0, bias=False)
		)
		self.modulation_act = get_act('sigmoid')()
		self.q = nn.Conv2d(dim, dim * 1, kernel_size=1, stride=1, padding=0, bias=qkv_bias)
		self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, stride=1, padding=0, bias=qkv_bias)
		self.proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj_drop = nn.Dropout(proj_drop)

	@torch.no_grad()
	def _get_ref_points(self, H, W, B, dtype, device):
		ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
									  torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device))
		ref = torch.stack((ref_y, ref_x), -1)
		ref[..., 1].div_(W).mul_(2).sub_(1)
		ref[..., 0].div_(H).mul_(2).sub_(1)
		ref = ref[None, ...].expand(B * self.d_groups, -1, -1, -1)  # B * g H W 2
		return ref

	def forward(self, x):
		B, C, H, W = x.shape
		q = self.q(x)
		q_off = rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.d_groups, c=self.n_group_dim).contiguous()
		offset_modulation = self.conv_offset_modulation(q_off)  # bg 3 h w
		offset, modulation = offset_modulation[:, 0:2, :, :], self.modulation_act(
			offset_modulation[:, 2:3, :, :])  # bg 2 h w, bg 1 h w
		H_off, W_off = offset.size(2), offset.size(3)

		offset_range = torch.tensor([1.0 / H_off, 1.0 / W_off], device=x.device).reshape(1, 2, 1, 1)
		offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
		offset = rearrange(offset, 'b c h w -> b h w c').contiguous()
		reference = self._get_ref_points(H_off, W_off, B, x.dtype, x.device)
		pos = offset + reference

		x_sampled = F.grid_sample(input=x.reshape(B * self.d_groups, self.n_group_dim, H, W),
								  grid=pos[..., (1, 0)],  # y, x -> x, y
								  mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
		x_sampled *= modulation.sigmoid()
		x_sampled = rearrange(x_sampled, '(b g) c h w -> b (g c) h w', b=B, g=self.d_groups).contiguous()
		q = rearrange(q, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head,
					  dim_head=self.dim_head).contiguous()
		kv = self.kv(x_sampled)
		kv = rearrange(kv, 'b (kv heads dim_head) h w -> kv b heads (h w) dim_head', kv=2, heads=self.num_head,
					   dim_head=self.dim_head).contiguous()
		k, v = kv[0], kv[1]

		attn = (q @ k.transpose(-2, -1)) * self.scale
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)

		x = attn @ v
		x = rearrange(x, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head,
					  dim_head=self.dim_head, h=H, w=W).contiguous()
		x = self.proj(x)
		x = self.proj_drop(x)
		return x


class DMSA_OP(nn.Module):

	def __init__(self, dim, dim_head, window_size, kernel_size, stride, qkv_bias=False, attn_drop=0., proj_drop=0.,
				 d_groups=3):
		super().__init__()
		self.window_size = window_size if isinstance(window_size, list) else [window_size, window_size]
		self.mdmsa = DMSA(dim, dim_head, kernel_size, stride, qkv_bias, attn_drop, proj_drop, d_groups)
		self.gamma_mdmsa = nn.Parameter(init_scale_values * torch.ones((dim)), requires_grad=True)

	def forward(self, x):
		B, C, H, W = x.shape
		if self.window_size[0] <= 0:
			window_size_W, window_size_H = W, H
		else:
			window_size_W, window_size_H = self.window_size[1], self.window_size[0]
		pad_l, pad_t = 0, 0
		pad_r = (window_size_W - W % window_size_W) % window_size_W
		pad_b = (window_size_H - H % window_size_H) % window_size_H
		x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))

		n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
		x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous()
		x = self.gamma_mdmsa.unsqueeze(0).unsqueeze(2).unsqueeze(3) * self.mdmsa(x)
		x = rearrange(x, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()

		if pad_r > 0 or pad_b > 0:
			x = x[:, :, :H, :W].contiguous()
		return x


class Conv_OP(nn.Module):

	def __init__(self, dim, kernel_size, stride=1):
		super().__init__()
		padding = math.ceil((kernel_size - stride) / 2)
		self.conv1 = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)
		self.norm1 = get_norm('bn_2d')(dim)
		self.act1 = get_act('silu')()
		self.conv2 = nn.Conv2d(dim, dim, 1, 1, 0)

	def forward(self, x):
		x = self.conv1(x)
		x = self.norm1(x)
		x = self.act1(x)
		x = self.conv2(x)
		return x


class DCN2_OP(nn.Module):
	# ref: https://github.com/WenmuZhou/DBNet.pytorch/blob/678b2ae55e018c6c16d5ac182558517a154a91ed/models/backbone/resnet.py
	def __init__(self, dim, kernel_size=3, stride=1, deform_groups=4):
		super().__init__()
		offset_channels = kernel_size * kernel_size * 2
		self.conv1_offset = nn.Conv2d(dim, deform_groups * offset_channels, kernel_size=3, stride=stride, padding=1)
		self.conv1 = DeformConv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
		self.norm1 = get_norm('bn_2d')(dim)
		self.act1 = get_act('silu')()
		self.conv2 = nn.Conv2d(dim, dim, 1, 1, 0)

	def forward(self, x):
		offset = self.conv1_offset(x)
		x = self.conv1(x, offset)
		x = self.norm1(x)
		x = self.act1(x)
		x = self.conv2(x)
		return x


class GLI(nn.Module):

	def __init__(self, in_dim, dim_head, window_size, kernel_size=5, qkv_bias=False, drop=0., attn_drop=0.,
				 drop_path=0., act_layer='gelu', norm_layer='bn_2d',
				 op_names=['msa', 'mdmsa', 'conv', 'dcn'], d_group=3, gli_split=False, gli_weight=True,
				 cr_ratio=None):
		super().__init__()
		self.op_names = op_names
		self.gli_split = gli_split
		self.gli_weight = gli_weight
		self.cr_ratio = cr_ratio
		self.op_num = len(op_names)
		self.norm = get_norm(norm_layer)(in_dim)
		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
		if self.op_num == 1:
			dims = [in_dim]
		else:
			if gli_split:
				if cr_ratio:
					assert self.op_num == 2
					dims = [int(in_dim * cr_ratio), round(in_dim * (1 - cr_ratio))]
				else:
					dim = in_dim // self.op_num
					assert dim * self.op_num == in_dim
					dims = [dim] * self.op_num
			else:
				dims = [in_dim] * self.op_num
		self.dims = dims
		self.ops = nn.ModuleList()
		for idx, op_name in enumerate(op_names):
			if op_name in ['conv', 'c']:
				op = Conv_OP(dims[idx], kernel_size, stride=1)
			elif op_name in ['dcn', 'dc']:
				op = DCN2_OP(dims[idx], kernel_size, stride=1, deform_groups=d_group)
			elif op_name in ['msa', 'm']:
				op = MSA_OP(dims[idx], dim_head, window_size, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
			elif op_name in ['mdmsa', 'dm']:
				op = DMSA_OP(dims[idx], dim_head, window_size, kernel_size=5, stride=1, qkv_bias=True,
							 attn_drop=attn_drop, proj_drop=drop, d_groups=d_group)
			else:
				raise 'invalid \'{}\' operation'.format(op_name)
			self.ops.append(op)
		if self.op_num > 1 and gli_weight:
			self.alphas = nn.Parameter(init_alpha_value * torch.ones(self.op_num), requires_grad=True)

	def forward(self, x):
		shortcut = x
		x = self.norm(x)
		if self.op_num == 1:
			x = self.ops[0](x)
		else:
			if self.gli_split:
				if self.cr_ratio:
					xs = [x[:, :self.dims[0], :, :], x[:, self.dims[0]:, :, :]]
				else:
					xs = torch.chunk(x, self.op_num, dim=1)
			else:
				xs = [x] * self.op_num
			if self.gli_weight:
				alphas = F.softmax(self.alphas, dim=-1)
				if self.gli_split:
					if self.cr_ratio:
						x = torch.cat([self.ops[i](xs[i]) * alphas[i] for i in range(self.op_num)], dim=1).contiguous()
					else:
						xs = torch.cat([self.ops[i](xs[i]).unsqueeze(dim=-1) * alphas[i] for i in range(self.op_num)],
									   dim=-1)
						x = rearrange(xs, 'b c h w n -> b (c n) h w').contiguous()
				else:
					xs = torch.cat([self.ops[i](xs[i]).unsqueeze(dim=-1) * alphas[i] for i in range(self.op_num)],
								   dim=-1)
					x = reduce(xs, 'b c h w n -> b c h w', 'mean').contiguous()
			else:
				if self.gli_split:
					x = torch.cat([self.ops[i](xs[i]) for i in range(self.op_num)], dim=1)
				else:
					xs = torch.cat([self.ops[i](xs[i]).unsqueeze(dim=-1) for i in range(self.op_num)], dim=-1)
					x = reduce(xs, 'b c h w n -> b c h w', 'mean').contiguous()
		x = shortcut + self.drop_path(x)
		return x


# ========== Multi-Scale Populations ==========
class MSP(nn.Module):

	def __init__(self, in_dim, emb_dim, kernel_size=3, c_group=-1, stride=1, dilations=[1, 2, 3], msra_mode='cat',
				 act_layer='silu', norm_layer='bn_2d', msra_weight=True):
		super().__init__()
		self.msra_mode = msra_mode
		self.msra_weight = msra_weight
		self.dilation_num = len(dilations)
		assert in_dim % c_group == 0
		c_group = (in_dim if c_group == -1 else c_group) if stride == 1 else 1
		self.convs = nn.ModuleList()
		for i in range(len(dilations)):
			padding = math.ceil(((kernel_size - 1) * dilations[i] + 1 - stride) / 2)
			self.convs.append(nn.Sequential(
				nn.Conv2d(in_dim, emb_dim, kernel_size, stride, padding, dilations[i], groups=c_group),
				get_act(act_layer)(emb_dim)))
		if self.dilation_num > 1 and msra_weight:
			self.alphas = nn.Parameter(init_alpha_value * torch.ones(self.dilation_num), requires_grad=True)
		self.conv_out = nn.Conv2d(emb_dim * (self.dilation_num if msra_mode == 'cat' else 1), emb_dim, kernel_size=1,
								  stride=1, padding=0, bias=False)

	def forward(self, x):
		# B, C, H, W
		if self.dilation_num == 1:
			x = self.convs[0](x)
		else:
			if self.msra_weight:
				alphas = F.softmax(self.alphas, dim=-1)
				x = torch.cat([self.convs[i](x).unsqueeze(dim=-1) * alphas[i] for i in range(self.dilation_num)],
							  dim=-1)
			else:
				x = torch.cat([self.convs[i](x).unsqueeze(dim=-1) for i in range(self.dilation_num)], dim=-1)
			if self.msra_mode == 'cat':
				x = rearrange(x, 'b c h w n -> b (c n) h w').contiguous()
			elif self.msra_mode == 'sum':
				x = reduce(x, 'b c h w n -> b c h w', 'mean').contiguous()
		x = self.conv_out(x)
		return x


class MSRA(nn.Module):

	def __init__(self, in_dim, emb_dim, kernel_size=3, c_group=-1, stride=1, dilations=[1, 2, 3], msra_mode='cat',
				 act_layer='silu', norm_layer='bn_2d', msra_weight=True, msra_norm=True, msra_skip=True, drop_path=0.):
		super().__init__()
		self.norm = get_norm(norm_layer)(in_dim) if msra_norm else nn.Identity()
		self.msp = MSP(in_dim, emb_dim, kernel_size, c_group, stride, dilations, msra_mode, act_layer, norm_layer,
					   msra_weight)
		self.msra_skip = msra_skip
		if msra_skip:
			if stride == 1:
				self.skip_conv = nn.Identity()
			else:
				self.skip_conv = nn.Sequential(
					nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
					nn.Conv2d(in_dim, emb_dim, 1, stride=1, padding=0, bias=False),
					get_norm(norm_layer)(emb_dim))
			self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

	def forward(self, x):
		shortcut = x
		x = self.msp(self.norm(x))
		if self.msra_skip:
			x = self.skip_conv(shortcut) + self.drop_path(x)
		return x


# ========== Block ==========
class EATBlock(nn.Module):

	def __init__(self, in_dim, emb_dim, kernel_size=3, stride=1, dilations=[1, 2, 3], norms=['bn_2d', 'bn_2d', 'bn_2d'],
				 msra_mode='cat', msra_weight=True, msra_norm=True, msra_skip=True,
				 dim_head=6, window_size=7, qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
				 op_names=['msa', 'conv'], d_group=3, c_group=-1, gli_split=False, gli_weight=True, mlp_ratio=4., ):
		super().__init__()
		self.layer1 = MSRA(in_dim, emb_dim, kernel_size, c_group, stride, dilations, msra_mode, 'silu',
									norms[0],
									msra_weight, msra_norm, msra_skip, drop_path)
		self.layer2 = GLI(emb_dim, dim_head, window_size, 5, qkv_bias,
								   drop, attn_drop, drop_path, 'silu', norms[1],
								   op_names, d_group, gli_split, gli_weight)
		self.layer3 = FFN(emb_dim, mlp_ratio, drop, drop_path, 'gelu', norms[2])

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		return x


# ========== Task-related Head ==========
class MCA(nn.Module):
	def __init__(self, dim, dim_head=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
		super().__init__()
		self.dim_head = dim_head
		self.num_head = dim // dim_head
		self.scale = self.dim_head ** -0.5

		self.q = nn.Conv2d(dim, dim * 1, kernel_size=1, stride=1, padding=0, bias=qkv_bias)
		self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, stride=1, padding=0, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, )
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, x, xq):
		B, C, H, W = x.shape
		_, _, Hq, Wq = xq.shape

		q = self.q(xq)
		kv = self.kv(x)
		q = rearrange(q, 'b (q heads dim_head) h w -> q b heads (h w) dim_head', q=1, heads=self.num_head,
					  dim_head=self.dim_head).contiguous()
		kv = rearrange(kv, 'b (kv heads dim_head) h w -> kv b heads (h w) dim_head', kv=2, heads=self.num_head,
					   dim_head=self.dim_head).contiguous()
		q, k, v = q[0], kv[0], kv[1]

		attn = (q @ k.transpose(-2, -1)) * self.scale
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)

		x = attn @ v
		x = rearrange(x, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_heads,
					  dim_head=self.dim_head, h=Hq, w=Wq).contiguous()
		x = self.proj(x)
		x = self.proj_drop(x)
		return x


class TRHead(nn.Module):

	def __init__(self, dim, dim_head, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
				 drop_path=0., act_layer='gelu', norm_layer='lnc'):
		super().__init__()
		self.norm_kv = get_norm(norm_layer)(dim)
		self.norm1 = get_norm(norm_layer)(dim)
		self.attn = MCA(dim, dim_head=dim_head, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
		self.norm2 = get_norm(norm_layer)(dim)
		hid_dim = int(dim * mlp_ratio)
		self.mlp = MLP(in_dim=dim, hid_dim=hid_dim, out_dim=dim, act_layer=act_layer, drop=drop)

	def forward(self, x, xq):
		xq = xq + self.drop_path(self.attn(self.norm_kv(x), self.norm1(xq)))
		xq = xq + self.drop_path(self.mlp(self.norm2(xq)))
		return xq


@BACKBONES.register_module()
class EAFormer(nn.Module):

	def __init__(self, in_dim=3,
				 depths=[2, 2, 6, 2], embed_dims=[64, 128, 256, 512], dim_heads=[32, 32, 32, 32],
				 window_sizes=[7, 7, 7, 7], kernel_sizes=[3, 3, 3, 3], down_mode='kernel',
				 dilations=[[1], [1], [1, 2, 3], [1, 2]], norms=['bn_2d', 'bn_2d', 'bn_2d'],
				 msra_mode='sum', msra_weight=True, msra_norm=True, msra_skip=True,
				 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
				 op_names=[['conv'], ['conv'], ['msa', 'conv'], ['msa', 'conv']],
				 d_groups=[3, 3, 3, 3], c_groups=[-1, -1, -1, -1], gli_split=False, gli_weight=True,
				 mlp_ratio=4., cls_head_nums=0, pretrained=None,
				 sync_bn=False, freeze_stages=-1, norm_eval=False, global_pers=[-1, -1, -1, -1]):
		super().__init__()
		dprs = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]  # stochastic depth decay rule
		self.stage0 = nn.ModuleList([
			MSRA(in_dim, embed_dims[0] // 2, kernel_size=3, c_group=1, stride=2, dilations=[1], msra_mode='sum',
						  act_layer='silu', norm_layer='bn_2d', msra_weight=False,
						  msra_norm=False, msra_skip=False),
			MSRA(embed_dims[0] // 2, embed_dims[0], kernel_size=3, c_group=1, stride=2, dilations=[1],
						  msra_mode='sum',
						  act_layer='silu', norm_layer='bn_2d', msra_weight=False,
						  msra_norm=True, msra_skip=False)
		])
		emb_dim_pre = embed_dims[0]
		for i in range(len(depths)):
			layers = []
			dpr = dprs[sum(depths[:i]):sum(depths[:i + 1])]
			for j in range(depths[i]):
				stride = 2 if j == 0 and i > 0 else 1
				kernel_size = stride if stride > 1 and down_mode == 'patch' else kernel_sizes[i]
				window_size = -1 if global_pers[i] > 0 and j % global_pers[i] == 0 else window_sizes[i]
				layers.append(EATBlock(emb_dim_pre, emb_dim=embed_dims[i], kernel_size=kernel_size, stride=stride,
									  dilations=dilations[i], norms=norms, msra_mode=msra_mode, msra_weight=msra_weight,
									  msra_norm=msra_norm,
									  msra_skip=msra_skip, dim_head=dim_heads[i], window_size=window_size,
									  qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=dpr[j],
									  op_names=op_names[i], d_group=d_groups[i], c_group=c_groups[i],
									  gli_split=gli_split, gli_weight=gli_weight, mlp_ratio=mlp_ratio, ))
				emb_dim_pre = embed_dims[i]
			self.__setattr__(f'stage{i + 1}', nn.ModuleList(layers))
		# cls head
		if cls_head_nums:
			self.cls_token = nn.Parameter(torch.zeros(1, embed_dims[-1], 1, 1))
			layers = [TRHead(embed_dims[-1], dim_heads[-1], mlp_ratio=mlp_ratio,
							 qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=0.)
					  for _ in range(cls_head_nums)]
			self.stage_cls = nn.ModuleList(layers)
		else:
			self.cls_token, self.stage_cls = None, nn.ModuleList()
		# linear head
		num_classes = 1000
		self.norm = nn.BatchNorm2d(embed_dims[-1])
		self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

		trunc_normal_(self.cls_token, std=.02) if self.cls_token is not None else None
		# self.apply(self.init_weights)

		self.freeze_stages = freeze_stages
		self.norm_eval = norm_eval
		# load pretrained model
		self.init_weights(pretrained)
		self._sync_bn() if sync_bn else None
		self._freeze_stages()

		# remove linear head for obj det/seg
		self.norm = None
		self.head = None
		self.cls_token = None
		self.stage_cls = None

	def init_weights(self, pretrained):
		if pretrained is None:
			for m in self.parameters():
				if isinstance(m, nn.Linear):
					trunc_normal_(m.weight, std=.02)
					if isinstance(m, nn.Linear) and m.bias is not None:
						nn.init.constant_(m.bias, 0)
				elif isinstance(m, nn.LayerNorm):
					nn.init.constant_(m.bias, 0)
					nn.init.constant_(m.weight, 1.0)
		else:
			state_dict = torch.load(pretrained, map_location='cpu')
			self_state_dict = self.state_dict()
			for k, v in state_dict.items():
				self_state_dict.update({k: v})
			self.load_state_dict(self_state_dict)

	def _sync_bn(self):
		self.stage0 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.stage0)
		self.stage1 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.stage1)
		self.stage2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.stage2)
		self.stage3 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.stage3)
		self.stage4 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.stage4)
		self.norm = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.norm)

	def _freeze_stages(self):
		for i in range(self.freeze_stages + 1):
			m = getattr(self, f'stage{i + 1}')
			m.eval()
			for param in m.parameters():
				param.requires_grad = False

	@torch.jit.ignore
	def no_weight_decay(self):
		return {'pos_embed', 'cls_token'}

	@torch.jit.ignore
	def no_weight_decay_keywords(self):
		return {'alpha', 'gamma', 'beta'}

	def get_classifier(self):
		return self.head

	def reset_classifier(self, num_classes, global_pool=''):
		self.num_classes = num_classes
		self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

	def forward_features(self, x):
		B, C, H, W = x.shape
		out = []

		for blk in self.stage0:
			x = blk(x)
		for blk in self.stage1:
			x = blk(x)
		out.append(x)
		for blk in self.stage2:
			x = blk(x)
		out.append(x)
		for blk in self.stage3:
			x = blk(x)
		out.append(x)
		for blk in self.stage4:
			x = blk(x)
		out.append(x)
		# if self.stage_cls:
		#     cls_token = self.cls_token.expand(B, -1, -1, -1)
		#     for blk in self.stage_cls:
		#         cls_token = blk(x, cls_token)
		#     x = cls_token

		return tuple(out)

	def forward(self, x):
		x1, x2, x3, x4 = self.forward_features(x)

		# x = self.norm(x)
		# x = reduce(x, 'b c h w -> b c', 'mean').contiguous()
		# x = self.head(x)

		return x1, x2, x3, x4

	def train(self, mode=True):
		"""Convert the model into training mode while keep normalization layer
		freezed."""
		super(EAFormer, self).train(mode)
		self._freeze_stages()
		if mode and self.norm_eval:
			for m in self.modules():
				# trick: eval have effect on BatchNorm only
				if isinstance(m, _BatchNorm):
					m.eval()
