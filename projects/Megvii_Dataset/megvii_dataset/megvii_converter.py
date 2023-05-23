# Copyright (c) OpenMMLab. All rights reserved.
import os
from os import path as osp
import mmengine
from pyquaternion import Quaternion
import json

class_names = [
    "小汽车", "汽车", "货车", "工程车", "巴士", "摩托车", "自行车", "三轮车", "骑车人", "骑行的人", "人",
    "行人", "其它", "残影", "蒙版", "其他", "拖挂", "锥桶", "防撞柱"
]


def create_megvii_infos(root_path, info_prefix):
    """处理旷视数据集的激光雷达数据
    root_path: data/kuangshi_data/ppl_bag_20220909_132234_det/
    ann_path: root_path/annotation_det/
    ann_path 下有10个 json文件 ，需要将10个 json 文件的数据解析出来放到一个 pkl 中，
    并把此pkl放到 root_path 目录下


    """

    metainfo = dict(class_names=class_names)

    train_infos, val_infos = _fill_trainval_infos(root_path)

    if train_infos is not None:
        data = dict(data_list=train_infos, metainfo=metainfo)
        info_path = osp.join(root_path,
                             '{}_infos_train.pkl'.format(info_prefix))
        mmengine.dump(data, info_path)

    if val_infos is not None:
        data['data_list'] = val_infos
        info_val_path = osp.join(root_path,
                                 '{}_infos_val.pkl'.format(info_prefix))
        mmengine.dump(data, info_val_path)


def _fill_trainval_infos(root_path):
    """全部划分为训练集
    """

    train_infos = []
    val_infos = None

    dir_fuser_lidar = os.path.join(root_path, 'fuser_lidar/')
    dir_annotation_det = os.path.join(root_path, "annotation_det/")
    json_files = os.listdir(dir_annotation_det)
    frame_idx = 0
    for json_file in json_files:
        json_path = os.path.join(dir_annotation_det, json_file)
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            for frame in json_data['frames']:
                if frame['is_key_frame']:
                    labels = frame['labels']
                    nori_id = frame['sensor_data']['fuser_lidar']['nori_id']
                    lidar_path = os.path.join(dir_fuser_lidar,
                                              nori_id.split(',')[0],
                                              nori_id + '.pcd')
                    info = dict(
                        sample_idx=frame_idx,
                        lidar_path=lidar_path,
                        instances=_get_instances(labels))
                    train_infos.append(info)
                    frame_idx += 1

    return train_infos, val_infos


def _get_instances(labels):
    """
    将旷视 labels 解析成 nuscenes instances 格式
    --------------------------------------------------
    旷视 labels：
    [
        {
            xyz_lidar:{'x':1, 'y':2, 'z':3},
            lwh:{'l':1,'w':2,'h':3},
            angle_lidar:{'w':1,'x':2,'y':3,'z':4}，
            category:'汽车'
        },
        {}, {}, ...

    ]
    --------------------------------------------------
    nuscenes instances：
    [
        {
            bbox_label_3d: 7
            bbox_3d: [x,y,z,l,w,h,r]
        },
        {}, {}, ...
    ]

    """

    instances = []
    for label in labels:
        ins = dict(
            bbox_label_3d=class_names.index(label['category']),  # 1
            bbox_3d=_label2box(label['xyz_lidar'], label['lwh'],
                               label['angle_lidar']))
        instances.append(ins)
    return instances


def _label2box(xyz_lidar, lwh, angle_lidar):
    """
    将参数中的三个字典，转换成 [x,y,z,l,w,h,r]
    """

    x = xyz_lidar['x']
    y = xyz_lidar['y']
    z = xyz_lidar['z']
    l = lwh['l']
    w = lwh['w']
    h = lwh['h']
    q = Quaternion(angle_lidar['w'], angle_lidar['x'], angle_lidar['y'],
                   angle_lidar['z'])
    bbox_3d = [x, y, z - h / 2, l, w, h, q.angle]

    return bbox_3d