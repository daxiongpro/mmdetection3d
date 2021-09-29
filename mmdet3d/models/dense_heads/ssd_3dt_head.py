import torch

from mmdet3d.models.builder import build_loss
from mmdet.models import HEADS
from .ssd_3d_head import SSD3DHead
from ...ops import furthest_point_sample


@HEADS.register_module()
class SSD3DTHead(SSD3DHead):
    r"""Bbox head of `3DSSD <https://arxiv.org/abs/2002.10187>`_.

    Args:
        num_classes (int): The number of class.
        bbox_coder (:obj:`BaseBBoxCoder`): Bbox coder for encoding and
            decoding boxes.
        in_channels (int): The number of input feature channel.
        train_cfg (dict): Config for training.
        test_cfg (dict): Config for testing.
        vote_module_cfg (dict): Config of VoteModule for point-wise votes.
        vote_aggregation_cfg (dict): Config of vote aggregation layer.
        pred_layer_cfg (dict): Config of classfication and regression
            prediction layers.
        conv_cfg (dict): Config of convolution in prediction layer.
        norm_cfg (dict): Config of BN in prediction layer.
        act_cfg (dict): Config of activation in prediction layer.
        objectness_loss (dict): Config of objectness loss.
        center_loss (dict): Config of center loss.
        dir_class_loss (dict): Config of direction classification loss.
        dir_res_loss (dict): Config of direction residual regression loss.
        size_res_loss (dict): Config of size residual regression loss.
        corner_loss (dict): Config of bbox corners regression loss.
        vote_loss (dict): Config of candidate points regression loss.
    """

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
                 act_cfg=dict(type='ReLU'),
                 objectness_loss=None,
                 center_loss=None,
                 dir_class_loss=None,
                 dir_res_loss=None,
                 size_res_loss=None,
                 corner_loss=None,
                 vote_loss=None,
                 init_cfg=None):
        super(SSD3DTHead, self).__init__(
            num_classes,
            bbox_coder,
            in_channels=in_channels,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            vote_module_cfg=vote_module_cfg,
            vote_aggregation_cfg=vote_aggregation_cfg,
            pred_layer_cfg=pred_layer_cfg,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            objectness_loss=objectness_loss,
            center_loss=center_loss,
            dir_class_loss=dir_class_loss,
            dir_res_loss=dir_res_loss,
            size_res_loss=size_res_loss,
            corner_loss=corner_loss,
            vote_loss=vote_loss,
            init_cfg=init_cfg)

        self.corner_loss = build_loss(corner_loss)
        self.vote_loss = build_loss(vote_loss)
        self.num_candidates = vote_module_cfg['num_points']

    def forward(self, feat_dict, sample_mod):
        """Forward pass.

        Note:
            The forward of VoteHead is devided into 4 steps:

                1. Generate vote_points from seed_points.
                2. Aggregate vote_points.
                3. Predict bbox and score.
                4. Decode predictions.

        Args:
            feat_dict (dict): Feature dict from backbone.
            sample_mod (str): Sample mode for vote aggregation layer.
                valid modes are "vote", "seed", "random" and "spec".

        Returns:
            dict: Predictions of vote head.
        """
        assert sample_mod in ['vote', 'seed', 'random', 'spec']

        seed_points, seed_features = feat_dict['sa_xyz'], feat_dict['sa_features']

        # 1. generate vote_points from seed_points
        vote_points, vote_features, vote_offset = self.vote_module(
            seed_points, seed_features)
        results = dict(
            seed_points=seed_points,
            vote_points=vote_points,
            vote_features=vote_features,
            vote_offset=vote_offset)

        # 2. aggregate vote_points
        if sample_mod == 'vote':
            # use fps in vote_aggregation
            aggregation_inputs = dict(
                points_xyz=vote_points, features=vote_features)
        elif sample_mod == 'seed':
            # FPS on seed and choose the votes corresponding to the seeds
            sample_indices = furthest_point_sample(seed_points,
                                                   self.num_proposal)
            aggregation_inputs = dict(
                points_xyz=vote_points,
                features=vote_features,
                indices=sample_indices)
        elif sample_mod == 'random':
            # Random sampling from the votes
            batch_size, num_seed = seed_points.shape[:2]
            sample_indices = seed_points.new_tensor(
                torch.randint(0, num_seed, (batch_size, self.num_proposal)),
                dtype=torch.int32)
            aggregation_inputs = dict(
                points_xyz=vote_points,
                features=vote_features,
                indices=sample_indices)
        elif sample_mod == 'spec':
            # Specify the new center in vote_aggregation
            aggregation_inputs = dict(
                points_xyz=seed_points,
                features=seed_features,
                target_xyz=vote_points)
        else:
            raise NotImplementedError(
                f'Sample mode {sample_mod} is not supported!')

        vote_aggregation_ret = self.vote_aggregation(**aggregation_inputs)
        aggregated_points, features, aggregated_indices = vote_aggregation_ret

        results['aggregated_points'] = aggregated_points
        results['aggregated_features'] = features
        results['aggregated_indices'] = aggregated_indices

        # 3. predict bbox and score
        cls_predictions, reg_predictions = self.conv_pred(features)

        # 4. decode predictions
        decode_res = self.bbox_coder.split_pred(cls_predictions,
                                                reg_predictions,
                                                aggregated_points)

        results.update(decode_res)

        return results
