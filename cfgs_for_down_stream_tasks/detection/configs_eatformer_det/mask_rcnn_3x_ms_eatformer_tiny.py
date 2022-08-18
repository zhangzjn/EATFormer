_base_ = [
    './_base_/models/mask_rcnn_eaformer_fpn.py',
    './_base_/datasets/coco_instance.py',
    './_base_/schedules/schedule_1x.py',
    './_base_/default_runtime.py'
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
    neck=dict(in_channels=[64, 128, 192, 256]))

optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[27, 33], warmup_iters=1, warmup_ratio=0.001,)
runner = dict(type='EpochBasedRunnerAmp', max_epochs=36)

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=dict(max_norm=5, norm_type=2), _delete_=True,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                 (736, 1333), (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333), (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

data = dict(
	samples_per_gpu=4,
	workers_per_gpu=4,
    train=dict(pipeline=train_pipeline),
)