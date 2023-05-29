import numpy as np
import torch
from mmengine.structures import InstanceData
from mmdet3d.structures import (DepthInstance3DBoxes, Det3DDataSample)
from mmdet3d.visualization import Det3DLocalVisualizer

import os
from pyntcloud import PyntCloud
import json
from pyquaternion import Quaternion


def labels2boxes(labels):
    box_list = []
    for box in labels:
        x = box['xyz_lidar']['x']
        y = box['xyz_lidar']['y']
        z = box['xyz_lidar']['z']
        l = box['lwh']['l']
        w = box['lwh']['w']
        h = box['lwh']['h']
        angle_lidar = box['angle_lidar']
        q = Quaternion(angle_lidar['w'], angle_lidar['x'], angle_lidar['y'],
                       angle_lidar['z'])
        box_np = np.array([[x, y, z - h / 2, l, w, h, q.angle]])
        box_list.append(box_np)
    box_list = box_list[:-1]  # 最后一个框是对向车道的蒙板
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


dir_ppl_bag_20220909_123325_det = "data/kuangshi_data/ppl_bag_20220909_123325_det/"
dir_annotation_det = dir_ppl_bag_20220909_123325_det + "annotation_det/"
dir_fuser_lidar = dir_ppl_bag_20220909_123325_det + 'fuser_lidar/'

json_files = os.listdir(dir_annotation_det)
for json_file in json_files:
    json_path = dir_annotation_det + json_file
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        for frame in json_data['frames']:
            if frame['is_key_frame']:
                labels = frame['labels']
                bbox = labels2boxes(labels)
                nori_id = frame['sensor_data']['fuser_lidar']['nori_id']
                pcdpath = dir_fuser_lidar + nori_id.split(
                    ',')[0] + '/' + nori_id + '.pcd'
                points = PyntCloud.from_file(pcdpath)
                show_data(points.xyz, bbox)
