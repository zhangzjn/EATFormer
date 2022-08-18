# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='EAFormer',
        pretrained=None,  # ckpt pretrained path
        in_dim=3,
        depths=[2, 2, 6, 2],
        embed_dims=[64, 128, 192, 256],
        dim_heads=[32, 32, 32, 32],
        window_sizes=[7, 7, 7, 7],
        kernel_sizes=[3, 3, 3, 3],
        down_mode='kernel',
        dilations=[[1], [1], [1, 2, 3], [1, 2]],
        norms=['bn_2d', 'bn_2d', 'bn_2d'],
        msra_mode='sum',
        msra_weight=True,
        msra_skip=True,
        qkv_bias=True,
        drop=0., attn_drop=0., drop_path=0.05,
        op_names=[['conv'], ['conv'], ['mdmsa', 'conv'], ['mdmsa', 'conv']],
        d_groups=[2, 2, 2, 2], c_groups=[-1, -1, -1, -1], gli_split=True, gli_weight=True,
        mlp_ratio=4., cls_head_nums=0,),
    decode_head=dict(
        type='UPerHead',
        in_channels=[64, 128, 192, 256],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=192,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))