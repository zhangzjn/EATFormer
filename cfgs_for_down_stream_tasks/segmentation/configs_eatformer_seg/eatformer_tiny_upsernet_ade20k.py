_base_ = [
    './_base_/models/upernet_eaformer.py',
    './_base_/datasets/ade20k.py',
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_160k.py'
]
model = dict(
	backbone=dict(
		pretrained="../pretrained_models/tiny.pth",
		depths=[2, 2, 6, 2], embed_dims=[64, 128, 192, 256], dim_heads=[32, 32, 32, 32], window_sizes=[14, 14, 14, 14],
		kernel_sizes=[3, 3, 3, 3], down_mode='kernel', dilations=[[1], [1], [1, 2, 3], [1, 2]],
		norms=['bn_2d', 'bn_2d', 'bn_2d'], msra_mode='sum', msra_weight=True, msra_skip=True,
		qkv_bias=True, drop=0., attn_drop=0., drop_path=0.05,
		op_names=[['conv'], ['conv'], ['mdmsa', 'conv'], ['mdmsa', 'conv']],
		d_groups=[2, 2, 2, 2], c_groups=[-1, -1, -1, -1], gli_split=True, gli_weight=True,
		mlp_ratio=4., cls_head_nums=2, sync_bn=True, freeze_stages=-1, norm_eval=False, global_pers=[-1, -1, 3, 2]
	),
    decode_head=dict(
        in_channels=[64, 128, 192, 256],
        num_classes=150
    ),
    auxiliary_head=dict(
        in_channels=192,  # aux heads
        num_classes=150
    ))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00012, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

data = dict(
	samples_per_gpu=4,
	workers_per_gpu=4)