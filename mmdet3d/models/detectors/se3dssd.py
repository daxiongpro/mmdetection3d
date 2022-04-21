# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS
from .single_stage import SingleStage3DDetector


@DETECTORS.register_module()
class SESSD3DNet_stu(SingleStage3DDetector):
    r"""`VoteNet <https://arxiv.org/pdf/1904.09664.pdf>`_ for 3D detection."""

    def __init__(self,
                 backbone,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(SESSD3DNet_stu, self).__init__(
            backbone=backbone,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=None,
            pretrained=pretrained)

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      pts_semantic_mask=None,
                      pts_instance_mask=None,
                      gt_bboxes_ignore=None):
        """Forward of training.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            img_metas (list): Image metas.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): gt bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): gt class labels of each batch.
            pts_semantic_mask (list[torch.Tensor]): point-wise semantic
                label of each batch.
            pts_instance_mask (list[torch.Tensor]): point-wise instance
                label of each batch.
            gt_bboxes_ignore (list[torch.Tensor]): Specify
                which bounding.

        Returns:
            dict: Losses.
        """
        # tea_preds = self.get_tea_preds(points, self.test_cfg.sample_mod)
        points_cat = torch.stack(points)

        x = self.extract_feat(points_cat)
        stu_preds = self.bbox_head(x, self.train_cfg.sample_mod)

        loss_inputs = (points, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask,
                       pts_instance_mask, img_metas)
        losses = self.bbox_head.loss(
            stu_preds, tea_preds, *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def get_tea_preds(self, points, sample_mod):
        # 新建一个SESSD3DNet_tea网络对象
        model_tea = copy.deepcopy(self)
        for param in model_tea.parameters():
            param.detach_()
        # 使用ema算法，将当前学生网络参数传递给教师网络
        # global_step = epoch * len(data_loader)
        # self.update_ema_variables(model_tea, global_step=None)
        tea_preds = model_tea.forward_test(points, None, sample_mod)


        # 得出预测值
        # 数据增强并反回
    def update_ema_variables(self, model_tea, global_step):
        """
        # 将学生网络（本网络）的参数使用ema算法传递给教师网络（model_tea）
        Args:
            model_tea:
            global_step:

        Returns:

        """

        alpha = min(1 - 1 / (global_step + 1), 0.999)
        for ema_param, param in zip(model_tea.parameters(), self.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Forward of testing.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list): Image metas.
            rescale (bool): Whether to rescale results.

        Returns:
            list: Predicted 3d boxes.
        """
        points_cat = torch.stack(points)

        x = self.extract_feat(points_cat)
        bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod)
        bbox_list = self.bbox_head.get_bboxes(
            points_cat, bbox_preds, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test with augmentation."""
        points_cat = [torch.stack(pts) for pts in points]
        feats = self.extract_feats(points_cat, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, pts_cat, img_meta in zip(feats, points_cat, img_metas):
            bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod)
            bbox_list = self.bbox_head.get_bboxes(
                pts_cat, bbox_preds, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]
