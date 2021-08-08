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

from mmdet.datasets import PIPELINES
from mmdet3d.datasets.pipelines import calibration


@PIPELINES.register_module()
class LoadPointsXY(object):
    """获取点云在图像上的对应xy坐标(B,N,2)

    Args:
        kwargs (dict): Arguments are the same as those in \
            :class:`LoadImageFromFile`.
    """

    def __init__(self):
        self.calib_dir = '/data/kitti/training'

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
        print(results['points'])
        pass
        sample_idx = results['sample_idx']
        calib = self.get_calib(sample_idx)
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)  # 点云在img上的坐标(N,2)，深度图的深度
        pts_origin_xy = pts_img[pts_valid_flag]  # 点云在img上的坐标
        results['xy'] = pts_origin_xy[choice, :]
        return results

    def _get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return calibration.Calibration(calib_file)
