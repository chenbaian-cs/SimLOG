_base_ = [
    '../_base_/datasets/sunrgbd-3d-10class.py', '../_base_/models/votenet.py', '../_base_/default_runtime.py'
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4
)
model = dict(
    backbone=dict(
        _delete_=True,
        type='DynamicPointInteraction',
        num_point=(2048, 1024, 512, 256),
        radii=(0.2, 0.4, 0.8, 1.2),
        num_sample=(16, 16, 16, 16),
        embed_dim=64,
        gmp_dim=64,
        res_expansion=0.5,
        use_xyz=True,
        use_pos_emb=True,
        ed_dims=(128, 256, 256, 512),
        fea_blocks=(1, 1, 1, 1),
        fp_dims=(256, 256)
        # type='DPISASSG',
        # in_channels=4,
        # num_points=(2048, 1024, 512, 256),
        # radius=(0.2, 0.4, 0.8, 1.2),
        # num_samples=(16, 16, 16, 16),
        # sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
        #           (128, 128, 256)),
        # fp_channels=((256, 256), (256, 256)),
        # norm_cfg=dict(type='BN2d'),
    ),
    bbox_head=dict(
        num_classes=10,
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=10,
            num_dir_bins=12,
            with_rot=True,
            mean_sizes=[
                [2.114256, 1.620300, 0.927272], [0.791118, 1.279516, 0.718182],
                [0.923508, 1.867419, 0.845495], [0.591958, 0.552978, 0.827272],
                [0.699104, 0.454178, 0.75625], [0.69519, 1.346299, 0.736364],
                [0.528526, 1.002642, 1.172878], [0.500618, 0.632163, 0.683424],
                [0.404671, 1.071108, 1.688889], [0.76584, 1.398258, 0.472728]
            ]
        )
    )
)
optimizer = dict(type='AdamW', lr=1e-3, weight_decay=5e-2)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[12, 24], gamma=0.2)
total_epochs = 36
evaluation = dict(interval=4)
