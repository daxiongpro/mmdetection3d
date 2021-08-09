#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   load_points_xy.py    
@Contact :   910660298@qq.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/3 15:33   daxiongpro    1.0         None
'''

# import lib
import os

from PIL import Image
from mmdet.datasets import PIPELINES
from mmdet3d.datasets.pipelines import calibration
import numpy as np


@PIPELINES.register_module()
class LoadPointsXY(object):
    """获取点云在图像上的对应xy坐标(B,N,2)

    Args:
        kwargs (dict): Arguments are the same as those in \
            :class:`LoadImageFromFile`.
    """

    def __init__(self, mode='training'):
        self.calib_dir = os.path.join('data/kitti/', mode, 'calib/')
        self.image_dir = ''

    def __call__(self, results):
        """

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
            dict_keys(['sample_idx', 'pts_filename', 'img_prefix', 'img_info', 'lidar2img', 'ann_info',
            'img_fields', 'bbox3d_fields', 'pts_mask_fields', 'pts_seg_fields',
            'bbox_fields', 'mask_fields', 'seg_fields', 'box_type_3d', 'box_mode_3d',
            'points', 'gt_bboxes_3d', 'gt_labels_3d', 'filename', 'ori_filename', 'img',
            'img_shape', 'ori_shape', 'gt_bboxes', 'gt_labels', 'scale', 'scale_idx',
            'pad_shape', 'scale_factor', 'keep_ratio', 'flip', 'flip_direction',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans'])
        Returns:
            dict: The dict contains loaded image and meta information.
        """

        sample_idx = results['sample_idx']
        calib = self.get_calib(sample_idx)
        pts_lidar = results['points']
        pts_lidar = pts_lidar.tensor.numpy()
        pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])  # 点云在相机坐标系下的坐标(N,3)
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)  # 点云在img上的坐标(N,2)，深度图的深度

        results['xy'] = pts_img
        return results

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return calibration.Calibration(calib_file)

    def _get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape):
        """
        返回是否为有效点
        Valid point should be in the image (and in the PC_AREA_SCOPE)
        :param pts_rect:点在相机坐标系下的坐标
        :param pts_img:点在图像坐标系下的坐标
        :param pts_rect_depth:点在相机坐标系下的深度
        :param img_shape:
        :return:[True, False, False, True, ...]
        """
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def _get_image_shape(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        im = Image.open(img_file)
        width, height = im.size
        return height, width, 3
