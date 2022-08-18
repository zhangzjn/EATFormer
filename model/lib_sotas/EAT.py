import math
from functools import partial

import torch
import torch.nn as nn
import numpy as np
# numpy-hilbert-curve==1.0.1
# pyzorder==0.0.1
from hilbert import decode, encode
from pyzorder import ZOrderIndexer

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from model import MODEL


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 256, 256), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim * 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, xq):
        B, N, C = x.shape
        _, Nq, _ = xq.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_ = self.q(xq).reshape(B, Nq, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q_[0], kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LocalAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., local_ks=3, length=196):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        mask = torch.ones(length, length)
        for i in range(length):
            for j in range(i-local_ks//2, i+local_ks//2+1, 1):
                j = min(max(0, j), length-1)
                mask[i, j] = 0
        mask = mask.unsqueeze(0).unsqueeze(1)
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill_(self.mask.bool(), -np.inf)
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LocalBranch(nn.Module):

    def __init__(self, dim, local_type='conv', local_ks=3, length=196, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.local_type = local_type
        if local_type == 'conv':
            self.linear = nn.Linear(dim, dim)
            self.local = nn.Conv1d(dim, dim, kernel_size=local_ks, padding=local_ks//2, padding_mode='zeros', groups=1)
        elif local_type == 'attn':
            self.local = LocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop,
                                        local_ks=local_ks, length=length)
        else:
            self.local = nn.Identity()

    def forward(self, x):
        if self.local_type in ['conv']:
            x = self.linear(x)
            x = x.permute(0, 2, 1)
            x = self.local(x)
            x = x.permute(0, 2, 1)
            return x
        elif self.local_type == 'attn':
            x = self.local(x)
            return x
        else:
            x = self.local(x)
            return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6), local_type='conv', local_ks=3, length=196, local_ratio=0.5, ffn_type='base'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if ffn_type == 'base':
            MLP = Mlp
        else:
            raise Exception('invalid ffn_type: {}'.format(ffn_type))
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Block_local(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), local_type='conv', local_ks=3, length=196, local_ratio=0.5, ffn_type='base'):
        super().__init__()
        local_dim = int(dim * local_ratio)
        self.global_dim = dim - local_dim
        div = 2
        self.num_heads = num_heads // div
        self.norm1 = norm_layer(self.global_dim)
        self.norm1_local = norm_layer(local_dim)
        self.attn = Attention(self.global_dim, num_heads=self.num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.local = LocalBranch(local_dim, local_type=local_type, local_ks=local_ks, length=length,
                                 num_heads=self.num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if ffn_type == 'base':
            MLP = Mlp
        else:
            raise Exception('invalid ffn_type: {}'.format(ffn_type))
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):  # torch.Size([64, 257, 192])
        x_attn = self.drop_path(self.attn(self.norm1(x[:, :, :self.global_dim])))
        x_local = self.drop_path(self.local(self.norm1_local(x[:, :, self.global_dim:])))
        x = x + torch.cat([x_attn, x_local], dim=2)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Block_cls(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6), local_type='conv', local_ks=3, local_ratio=0.5, ffn_type='base'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if ffn_type == 'base':
            MLP = Mlp
        else:
            raise Exception('invalid ffn_type: {}'.format(ffn_type))
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, xq):
        xq = xq + self.drop_path(self.attn(x, self.norm1(xq)))
        xq = xq + self.drop_path(self.mlp(self.norm2(xq)))
        return xq


class LocED(nn.Module):

    def __init__(self, size=16, size_p=1, dim=2, loc_encoder='sis'):
        super().__init__()
        size = int(size)
        if loc_encoder in ['zorder', 'hilbert']:
            if size & (size - 1) != 0:
                raise 'invalid size \'{}\' for \'{}\' mode'.format(size, loc_encoder)
        if loc_encoder in ['sis']:
            if size_p == 1:
                raise 'invalid size \'{}\' for \'{}\' mode'.format(size_p, loc_encoder)
        max_num = size ** dim
        indexes = np.arange(max_num)
        if 'sweep' == loc_encoder:  # ['sweep', 'scan', 'zorder', 'hilbert', 'sis']
            locs_flat = indexes
        elif 'scan' == loc_encoder:
            indexes = indexes.reshape(size, size)
            for i in np.arange(1, size, step=2):
                indexes[i, :] = indexes[i, :][::-1]
            locs_flat = indexes.reshape(-1)
        elif 'zorder' == loc_encoder:
            zi = ZOrderIndexer((0, size - 1), (0, size - 1))
            locs_flat = []
            for z in indexes:
                r, c = zi.rc(int(z))
                locs_flat.append(c * size + r)
            locs_flat = np.array(locs_flat)
        elif 'hilbert' == loc_encoder:
            bit = int(math.log2(size))
            locs = decode(indexes, dim, bit)
            locs_flat = self.flat_locs_hilbert(locs, dim, bit)
        elif 'sis' == loc_encoder:
            locs_flat = []
            axis_patches = size // size_p
            for i in range(axis_patches):
                for j in range(axis_patches):
                    for ii in range(size_p):
                        for jj in range(size_p):
                            locs_flat.append((size_p * i + ii) * size + (size_p * j + jj))
            locs_flat = np.array(locs_flat)
        else:
            raise Exception('invalid encoder mode')
        locs_flat_inv = np.argsort(locs_flat)
        index_flat = torch.LongTensor(locs_flat.astype(np.int64)).unsqueeze(0).unsqueeze(2)
        index_flat_inv = torch.LongTensor(locs_flat_inv.astype(np.int64)).unsqueeze(0).unsqueeze(2)
        self.index_flat = nn.Parameter(index_flat, requires_grad=False)
        self.index_flat_inv = nn.Parameter(index_flat_inv, requires_grad=False)

    def flat_locs_hilbert(self, locs, num_dim, num_bit):
        ret = []
        l = 2 ** num_bit
        for i in range(len(locs)):
            loc = locs[i]
            loc_flat = 0
            for j in range(num_dim):
                loc_flat += loc[j] * (l ** j)
            ret.append(loc_flat)
        return np.array(ret).astype(np.uint64)

    def __call__(self, img):
        img_encode = self.encode(img)
        return img_encode

    def encode(self, img):
        img_encode = torch.zeros(img.shape, dtype=img.dtype, device=img.device).scatter_(1, self.index_flat_inv.expand(img.shape), img)
        return img_encode

    def decode(self, img):
        img_decode = torch.zeros(img.shape, dtype=img.dtype, device=img.device).scatter_(1, self.index_flat.expand(img.shape), img)
        return img_decode


class EATransformer(nn.Module):

    def __init__(self, img_size=256, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, depth_cls=2,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 pos_emb=True, cls_token=False, cls_token_head=True, loc_encoder='sis', block_type='base_local', local_type='conv',
                 local_ks=3, local_ratio=0.5, ffn_type='base', sfc_mode='first'):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.cls_token_ = cls_token
        self.cls_token_head_ = cls_token_head
        self.sfc_mode = sfc_mode

        axis_patches = img_size // patch_size
        num_patches = axis_patches ** 2
        self.num_patches = num_patches
        if sfc_mode == 'first':
            self.loc_encoder = LocED(size=img_size, size_p=patch_size, dim=2, loc_encoder=loc_encoder)
            self.patch_embed = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size ** 2, stride=patch_size ** 2)
        elif sfc_mode == 'second':
            self.loc_encoder = LocED(size=axis_patches, size_p=1, dim=2, loc_encoder=loc_encoder)
            self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            raise 'invalid sfc_mode: {}'.format(sfc_mode)

        self.pos_drop = nn.Dropout(p=drop_rate)
        # body
        if block_type == 'base':
            BLOCK = Block
        elif block_type == 'base_local':
            BLOCK = Block_local
        else:
            raise Exception('invalid block_type: {}'.format(block_type))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = [BLOCK(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[i], norm_layer=norm_layer, local_type=local_type, local_ks=local_ks, length=num_patches, local_ratio=local_ratio, ffn_type=ffn_type)
                  for i in range(depth)]
        self.blocks = nn.ModuleList(blocks)
        # head
        if cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.blocks_cls = None
        elif cls_token_head:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            blocks_cls = [
                Block_cls(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                          drop_path=0, norm_layer=norm_layer, local_type=local_type, local_ks=local_ks, ffn_type=ffn_type)
                for _ in range(depth_cls)]
            self.blocks_cls = nn.ModuleList(blocks_cls)
        else:
            self.cls_token = None
            self.blocks_cls = None
            self.gap = nn.AdaptiveAvgPool1d(1)
        if pos_emb:
            if cls_token:
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
            else:
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            self.pos_embed = None
        self.norm = norm_layer(embed_dim)
        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02) if self.pos_embed is not None else None
        trunc_normal_(self.cls_token, std=.02) if self.cls_token is not None else None
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        if self.sfc_mode == 'first':
            x = self.loc_encoder.encode(x.flatten(2).transpose(1, 2))
            x = self.patch_embed(x.transpose(1, 2)).transpose(1, 2)  # torch.Size([2, 256, 768])
        elif self.sfc_mode == 'second':
            x = self.patch_embed(x).flatten(2).transpose(1, 2)
            x = self.loc_encoder.encode(x).contiguous()
        if self.cls_token_:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        if self.cls_token_:
            out = x[:, 0]
        elif self.cls_token_head_:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            for block in self.blocks_cls:
                cls_tokens = block(x, cls_tokens)
            out = cls_tokens[:, 0]
        else:
            out = self.gap(x.permute(0, 2, 1))[:, :, 0]
        return out

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# tiny | small | base in 224
@MODEL.register_module
def eat_tiny_patch16_224(pretrained=False, **kwargs):
    model = EATransformer(img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                          **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(url="", map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


@MODEL.register_module
def eat_small_patch16_224(pretrained=False, **kwargs):
    model = EATransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                          **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(url="", map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


@MODEL.register_module
def eat_base_patch16_224(pretrained=False, **kwargs):
    model = EATransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                          **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(url="", map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model
