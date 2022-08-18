_base_ = [
    './_base_/models/mask_rcnn_eaformer_fpn.py',
    './_base_/datasets/coco_instance.py',
    './_base_/schedules/schedule_1x.py',
    './_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        pretrained="../pretrained_models/small.pth",
		depths=[3, 4, 12, 3], embed_dims=[64, 128, 320, 448], dim_heads=[32, 32, 32, 32], window_sizes=[14, 14, 14, 14],
		kernel_sizes=[3, 3, 3, 3], down_mode='kernel', dilations=[[1], [1], [1, 2, 3], [1, 2]],
		norms=['bn_2d', 'bn_2d', 'bn_2d'], msra_mode='sum', msra_weight=True, msra_skip=True,
		qkv_bias=True, drop=0., attn_drop=0., drop_path=0.10,
		op_names=[['conv'], ['conv'], ['mdmsa', 'conv'], ['mdmsa', 'conv']],
		d_groups=[2, 2, 2, 2], c_groups=[-1, -1, -1, -1], gli_split=True, gli_weight=True,
		mlp_ratio=4., cls_head_nums=0, sync_bn=True, freeze_stages=-1, norm_eval=False, global_pers=[-1, -1, -1, -1]
    ),
    neck=dict(in_channels=[64, 128, 320, 448]))

optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[8, 11])
runner = dict(type='EpochBasedRunnerAmp', max_epochs=12)

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)

data = dict(
	samples_per_gpu=4,
	workers_per_gpu=4)