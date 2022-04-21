# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops.nms import batched_nms
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.core.bbox.structures import (DepthInstance3DBoxes,
                                          LiDARInstance3DBoxes,
                                          rotation_3d_in_axis)
from mmdet3d.models.builder import build_loss
from mmdet.core import multi_apply
from mmdet.models import HEADS
from .vote_head import VoteHead


@HEADS.register_module()
class SESSD3DHead(VoteHead):

    def __init__(self,
                 num_classes,
                 bbox_coder,
                 in_channels=256,
                 train_cfg=None,
                 test_cfg=None,
                 vote_module_cfg=None,
                 vote_aggregation_cfg=None,
                 pred_layer_cfg=None,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 ce_loss=None,
                 od_iou_loss=None,
                 init_cfg=None):
        super(SESSD3DHead, self).__init__(
            num_classes,
            bbox_coder,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            vote_module_cfg=vote_module_cfg,
            vote_aggregation_cfg=vote_aggregation_cfg,
            pred_layer_cfg=pred_layer_cfg,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            objectness_loss=None,
            center_loss=None,
            dir_class_loss=None,
            dir_res_loss=None,
            size_class_loss=None,
            size_res_loss=None,
            semantic_loss=None,
            init_cfg=init_cfg)

        self.ce_loss = build_loss(ce_loss)
        self.od_iou_loss = build_loss(od_iou_loss)
        self.num_candidates = vote_module_cfg['num_points']

    def loss(self, stu_preds, tea_preds, gt):
        """
        计算ce_loss 和 od_iou_loss
        Args:
            stu_preds: 学生预测结果
            tea_preds: 教师预测结果
            gt: ground truth
        Returns:
        """

        tea_preds = self.aug_tea_preds(tea_preds)  # 教师预测结果数据增强
        ce_loss = self.ce_loss(stu_preds, tea_preds)
        od_iou_loss = self.od_iou_loss(stu_preds, gt)
        losses = dict(
            ce_loss=ce_loss,
            od_iou_loss=od_iou_loss
        )

        return losses

    def aug_tea_preds(self, tea_preds):
        #  todo
        pass

    def get_bboxes(self, points, bbox_preds, input_metas, rescale=False):
        """Generate bboxes from 3DSSD head predictions.

        Args:
            points (torch.Tensor): Input points.
            bbox_preds (dict): Predictions from sdd3d head.
            input_metas (list[dict]): Point cloud and image's meta info.
            rescale (bool): Whether to rescale bboxes.

        Returns:
            list[tuple[torch.Tensor]]: Bounding boxes, scores and labels.
        """
        # decode boxes
        sem_scores = F.sigmoid(bbox_preds['obj_scores']).transpose(1, 2)
        obj_scores = sem_scores.max(-1)[0]
        bbox3d = self.bbox_coder.decode(bbox_preds)

        batch_size = bbox3d.shape[0]
        results = list()

        for b in range(batch_size):
            bbox_selected, score_selected, labels = self.multiclass_nms_single(
                obj_scores[b], sem_scores[b], bbox3d[b], points[b, ..., :3],
                input_metas[b])

            bbox = input_metas[b]['box_type_3d'](
                bbox_selected.clone(),
                box_dim=bbox_selected.shape[-1],
                with_yaw=self.bbox_coder.with_rot)
            results.append((bbox, score_selected, labels))

        return results

    def multiclass_nms_single(self, obj_scores, sem_scores, bbox, points,
                              input_meta):
        """Multi-class nms in single batch.

        Args:
            obj_scores (torch.Tensor): Objectness score of bounding boxes.
            sem_scores (torch.Tensor): Semantic class score of bounding boxes.
            bbox (torch.Tensor): Predicted bounding boxes.
            points (torch.Tensor): Input points.
            input_meta (dict): Point cloud and image's meta info.

        Returns:
            tuple[torch.Tensor]: Bounding boxes, scores and labels.
        """
        bbox = input_meta['box_type_3d'](
            bbox.clone(),
            box_dim=bbox.shape[-1],
            with_yaw=self.bbox_coder.with_rot,
            origin=(0.5, 0.5, 0.5))

        if isinstance(bbox, (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            box_indices = bbox.points_in_boxes_all(points)
            nonempty_box_mask = box_indices.T.sum(1) >= 0
        else:
            raise NotImplementedError('Unsupported bbox type!')

        corner3d = bbox.corners
        minmax_box3d = corner3d.new(torch.Size((corner3d.shape[0], 6)))
        minmax_box3d[:, :3] = torch.min(corner3d, dim=1)[0]
        minmax_box3d[:, 3:] = torch.max(corner3d, dim=1)[0]

        bbox_classes = torch.argmax(sem_scores, -1)
        nms_keep = batched_nms(
            minmax_box3d[nonempty_box_mask][:, [0, 1, 3, 4]],
            obj_scores[nonempty_box_mask], bbox_classes[nonempty_box_mask],
            self.test_cfg.nms_cfg)[1]

        if nms_keep.shape[0] > self.test_cfg.max_output_num:
            nms_keep = nms_keep[:self.test_cfg.max_output_num]

        # filter empty boxes and boxes with low score
        scores_mask = (obj_scores >= self.test_cfg.score_thr)
        nonempty_box_inds = torch.nonzero(
            nonempty_box_mask, as_tuple=False).flatten()
        nonempty_mask = torch.zeros_like(bbox_classes).scatter(
            0, nonempty_box_inds[nms_keep], 1)
        selected = (nonempty_mask.bool() & scores_mask.bool())

        if self.test_cfg.per_class_proposal:
            bbox_selected, score_selected, labels = [], [], []
            for k in range(sem_scores.shape[-1]):
                bbox_selected.append(bbox[selected].tensor)
                score_selected.append(obj_scores[selected])
                labels.append(
                    torch.zeros_like(bbox_classes[selected]).fill_(k))
            bbox_selected = torch.cat(bbox_selected, 0)
            score_selected = torch.cat(score_selected, 0)
            labels = torch.cat(labels, 0)
        else:
            bbox_selected = bbox[selected].tensor
            score_selected = obj_scores[selected]
            labels = bbox_classes[selected]

        return bbox_selected, score_selected, labels

    def _assign_targets_by_points_inside(self, bboxes_3d, points):
        """Compute assignment by checking whether point is inside bbox.

        Args:
            bboxes_3d (BaseInstance3DBoxes): Instance of bounding boxes.
            points (torch.Tensor): Points of a batch.

        Returns:
            tuple[torch.Tensor]: Flags indicating whether each point is
                inside bbox and the index of box where each point are in.
        """
        if isinstance(bboxes_3d, (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            points_mask = bboxes_3d.points_in_boxes_all(points)
            assignment = points_mask.argmax(dim=-1)
        else:
            raise NotImplementedError('Unsupported bbox type!')

        return points_mask, assignment
