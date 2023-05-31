import numpy as np
import torch
from mmengine.structures import InstanceData
from mmdet3d.structures import (DepthInstance3DBoxes, Det3DDataSample)
from mmdet3d.visualization import Det3DLocalVisualizer

import os
import os.path as osp
from pyntcloud import PyntCloud
import json
from pyquaternion import Quaternion


def bboxes3D_list2boxes(bboxes3D_list):
    box_list = []
    for box in bboxes3D_list:
        x, y, z = box["relativePos"]
        l, w, h = box["size"]
        yaw, pitch, roll = box['relativeRot']
        box_np = np.array([[x, y, z - h / 2, w, l, h, pitch]])
        box_list.append(box_np)
    boxes_np = np.concatenate(box_list, axis=0)
    return boxes_np


def show_data(points=np.random.rand(1000, 3), bbox=torch.rand((5, 7))):
    """show point cloud data with openmmlab and open3d.

    :param points: point clouds, defaults to np.random.rand(1000, 3)
    :type points: numpy ndarray, optional
    :param bbox: bounding boxes, (xzywlhr), defaults to torch.rand((5, 7))
    :type bbox: numpy ndarray, optional
    """
    det3d_local_visualizer = Det3DLocalVisualizer()

    # points = np.random.rand(1000, 3)

    gt_instances_3d = InstanceData()
    gt_instances_3d.bboxes_3d = DepthInstance3DBoxes(bbox)
    # gt_instances_3d.labels_3d = torch.randint(0, 2, (5, ))

    gt_det3d_data_sample = Det3DDataSample()
    gt_det3d_data_sample.gt_instances_3d = gt_instances_3d

    data_input = dict(points=points)

    det3d_local_visualizer.add_datasample(
        '3D Scene',
        data_input,
        gt_det3d_data_sample,
        vis_task='lidar_det',
        show=True)


dataset_root = 'data/meta_dataset/data_2023_05_30'
dir_json = osp.join(dataset_root, 'fuser_lidar')
dir_fuser_lidar = osp.join(dataset_root, 'fuser_lidar')
# dir_fuser_lidar = osp.join(dataset_root, 'scenes/0002/fuser_lidar')

# pcd_extension = '.pcd'
json_extension = '.json'  # 替换为你想要的文件后缀

# fuser_lidar_list = [file for file in os.listdir(dir_fuser_lidar) if file.endswith(pcd_extension)]
# json_list = [file for file in os.listdir(dir_json) if file.endswith(json_extension)]

json_files = [
    file for file in os.listdir(dir_json) if file.endswith(json_extension)
]
for json_file in json_files:
    json_path = osp.join(dir_json, json_file)
    id_ = json_file.split(".")[0]
    pcd_path = osp.join(dir_fuser_lidar, id_ + '.pcd')
    # pcd_path = osp.join(dir_fuser_lidar, id_ + '.pcd.bin')
    with open(json_path, 'r') as f:
        json_data = json.load(f)

        bboxes3D_list = json_data['bboxes3D']
        bbox = bboxes3D_list2boxes(bboxes3D_list)

        points = PyntCloud.from_file(pcd_path)
        # points = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 5)
        # xyz = points[:, :3]
        show_data(points.xyz, bbox)
        # show_data(xyz, bbox)
