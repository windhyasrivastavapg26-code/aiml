exp_name = 'zeniqa_learnable_prompt'
model = dict(
    type='CLIPIQA',
    generator=dict(
        type='CLIPIQALearnable',
        backbone_name='RN50',
        init_std=0.0,
        pos_embedding=False,
        classnames=[['sharp photo, good photo.', 'blurred photo, bad photo.'],
                    [
                        'noiseless photo, good photo.',
                        'noise photo, bad photo.'
                    ],
                    [
                        'well exposed photo, good photo.',
                        'white out overexposed photo, bad photo.'
                    ],
                    [
                        'well exposed photo, good photo.',
                        'black out underexposed photo, bad photo.'
                    ]]),
    pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'))
train_cfg = None
test_cfg = dict(metrics=['L1DIS'], crop_border=0)
train_dataset_type = 'IQAKoniqDataset'
val_dataset_type = 'IQAKoniqDataset'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(
        type='Normalize',
        keys=['lq'],
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
        to_rgb=True),
    dict(type='Flip', keys=['lq'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['lq'], flip_ratio=0.5, direction='vertical'),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path']),
    dict(type='ImageToTensor', keys=['lq']),
    dict(type='ToTensor', keys=['gt'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged',
        backend='pillow'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(
        type='Normalize',
        keys=['lq'],
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
        to_rgb=True),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path']),
    dict(type='ImageToTensor', keys=['lq']),
    dict(type='ToTensor', keys=['gt'])
]
data = dict(
    workers_per_gpu=6,
    train_dataloader=dict(samples_per_gpu=16, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=100,
        dataset=dict(
            type='IQAKoniqDataset',
            img_folder='/root/4T/dataset/koniq10k/1024x768/',
            ann_file=
            '/root/4T/dataset/koniq10k/koniq10k_distributions_sets.csv',
            pipeline=[
                dict(
                    type='LoadImageFromFile',
                    io_backend='disk',
                    key='lq',
                    flag='unchanged'),
                dict(type='RescaleToZeroOne', keys=['lq']),
                dict(
                    type='Normalize',
                    keys=['lq'],
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                    to_rgb=True),
                dict(
                    type='Flip',
                    keys=['lq'],
                    flip_ratio=0.5,
                    direction='horizontal'),
                dict(
                    type='Flip',
                    keys=['lq'],
                    flip_ratio=0.5,
                    direction='vertical'),
                dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path']),
                dict(type='ImageToTensor', keys=['lq']),
                dict(type='ToTensor', keys=['gt'])
            ],
            test_mode=False)),
    val=dict(
        type='IQAKoniqDataset',
        img_folder='/root/4T/dataset/koniq10k/1024x768/',
        ann_file='/root/4T/dataset/koniq10k/koniq10k_distributions_sets.csv',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='lq',
                flag='unchanged',
                backend='pillow'),
            dict(type='RescaleToZeroOne', keys=['lq']),
            dict(
                type='Normalize',
                keys=['lq'],
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
                to_rgb=True),
            dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path']),
            dict(type='ImageToTensor', keys=['lq']),
            dict(type='ToTensor', keys=['gt'])
        ],
        test_mode=True),
    test=dict(
        type='IQAKoniqDataset',
        img_folder='/root/4T/dataset/koniq10k/1024x768/',
        ann_file='/root/4T/dataset/koniq10k/koniq10k_distributions_sets.csv',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='lq',
                flag='unchanged',
                backend='pillow'),
            dict(type='RescaleToZeroOne', keys=['lq']),
            dict(
                type='Normalize',
                keys=['lq'],
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
                to_rgb=True),
            dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path']),
            dict(type='ImageToTensor', keys=['lq']),
            dict(type='ToTensor', keys=['gt'])
        ],
        test_mode=True))
optimizers = dict(
    generator=dict(
        type='AdamW', lr=0.0002, weight_decay=0.0001, betas=(0.9, 0.999)))
total_iters = 30000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[30000],
    restart_weights=[1],
    min_lr=1e-07)
checkpoint_config = dict(interval=2000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=2000, save_image=False, gpu_collect=True)
log_config = dict(
    interval=100, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
visual_config = None
dist_params = dict(backend='nccl', port='29515')
log_level = 'INFO'
work_dir = './work_dirs/zeniqa_learnable_prompt'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = False
