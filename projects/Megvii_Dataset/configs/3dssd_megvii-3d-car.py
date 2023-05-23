_base_ = [
    # 'mmdet3d::_base_/default_runtime.py',
    '../../../configs/3dssd/3dssd_4xb4_kitti-3d-car.py'
]

custom_imports = dict(
    imports=[
        'projects.Megvii_Dataset.megvii_dataset.megvii_dataset',
        'projects.Megvii_Dataset.megvii_dataset.loading'
    ],
    allow_failed_imports=False)


model = dict(
    backbone=dict(
        in_channels=3),
    bbox_head=dict(
        num_classes=19,
        bbox_coder=dict(
            type='AnchorFreeBBoxCoder', num_dir_bins=12, with_rot=True)))


# dataset settings
dataset_type = 'MegviiDataset'
ann_file = 'megvii_infos_train.pkl'
data_root = 'data/kuangshi_data/ppl_bag_20220909_132234_det/'

class_names = [
    "小汽车", "汽车", "货车", "工程车", "巴士", "摩托车", "自行车", "三轮车", "骑车人", "骑行的人", "人",
    "行人", "其它", "残影", "蒙版", "其他", "拖挂", "锥桶", "防撞柱"
]

point_cloud_range = [0, -40, -5, 70, 40, 3]
db_sampler = dict(_delete_=True)
val_evaluator = dict(type='MegviiMetric', ann_file=data_root + ann_file)
test_evaluator = dict(type='MegviiMetric', ann_file=data_root + ann_file)
metainfo = dict(_delete_=True, class_names=class_names)

train_pipeline = [
    dict(
        type='MegviiLoadPointsFromFile',
        # type='LoadPointsFromFile',
        coord_type='LIDAR',
        use_dim=3),
    dict(
        type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0],
        global_rot_range=[0.0, 0.0],
        rot_range=[-1.0471975511965976, 1.0471975511965976]),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.9, 1.1]),
    # 3DSSD can get a higher performance without this transform
    # dict(type='BackgroundPointsFilter', bbox_enlarge_range=(0.5, 2.0, 0.5)),
    dict(type='PointSample', num_points=16384),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='MegviiLoadPointsFromFile', coord_type='LIDAR', load_dim=3, use_dim=3),
    # dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=3, use_dim=3),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='PointSample', num_points=16384),
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=ann_file,
            pipeline=train_pipeline,
            metainfo=metainfo)))
test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        pipeline=test_pipeline,
        metainfo=metainfo))
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        pipeline=test_pipeline,
        metainfo=metainfo))

# test_dataloader = dict(_delete_=True)
# val_dataloader = dict(_delete_=True)
