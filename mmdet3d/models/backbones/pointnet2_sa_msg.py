from typing import List

import torch
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16
from torch import nn as nn

from mmdet3d.ops import build_sa_module
from mmdet.models import BACKBONES
from .base_pointnet import BasePointNet
from torch.nn.functional import grid_sample
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, outplanes, stride)
        self.bn1 = BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(outplanes, outplanes, 2 * stride)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out


# ================addition attention (add)=======================#
class ImageAttentionLayer(nn.Module):
    # image-attention 层。由图片和点云生成权值，乘到img特征上
    def __init__(self, channels):
        super(ImageAttentionLayer, self).__init__()
        self.ic, self.pc = channels  # 图像通道数，点云通道数
        rc = self.pc // 4  # 统一成rc通道
        self.conv1 = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),
                                   nn.BatchNorm1d(self.pc),
                                   nn.ReLU())
        self.fc1 = nn.Linear(self.ic, rc)
        self.fc2 = nn.Linear(self.pc, rc)
        self.fc3 = nn.Linear(rc, 1)

    def forward(self, img_feas, point_feas):
        """
        由图片和点云生成权值，乘到img特征上
        @param img_feas: 图片特征
        @param point_feas:点云特征
        @return: 带w权值的图片特征
        """
        batch = img_feas.size(0)
        img_feas_f = img_feas.transpose(1, 2).contiguous().view(-1, self.ic)  # BCN->BNC->(BN)C
        point_feas_f = point_feas.transpose(1, 2).contiguous().view(-1, self.pc)  # BCN->BNC->(BN)C'
        ri = self.fc1(img_feas_f)
        rp = self.fc2(point_feas_f)
        # att = F.sigmoid(self.fc3(F.tanh(ri + rp)))  # BNx1
        att = torch.sigmoid(self.fc3(torch.tanh(ri + rp)))  # BNx1
        att = att.squeeze(1)
        att = att.view(batch, 1, -1)  # B1N
        img_feas_new = self.conv1(img_feas)
        out = img_feas_new * att
        return out


class AttenFusionConv(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes):
        """"
        inplanes_I:输入的img 通道数
        inplanes_P:输入的point 通道数
        outplanes：输出的point 通道数
        """
        super(AttenFusionConv, self).__init__()
        self.IA_Layer = ImageAttentionLayer(channels=[inplanes_I, inplanes_P])
        self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_P, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, point_features, img_features):
        """
        融合模块
        @param point_features: (B, C1, N)
        @param img_features: (B, C2, N)
        @return:
        """
        img_features = self.IA_Layer(img_features, point_features)
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))
        return fusion_features


def feature_gather(feature_map, xy):
    """获取feature_map上 xy点的特征
    :param xy:(B,M,2)  normalize to [-1,1]
    :param feature_map:(B,C,H,W)
    :return:
    """
    # xy(B,M,2)->(B,1,M,2)
    xy = xy.unsqueeze(1)
    interpolate_feature = grid_sample(feature_map, xy, align_corners=True)  # (B,C,1,M)
    return interpolate_feature.squeeze(2)  # (B,C,M)


@BACKBONES.register_module()
class PointNet2SAMSG(BasePointNet):
    """PointNet2 with Multi-scale grouping.

    Args:
        in_channels (int): Input channels of point cloud.
        num_points (tuple[int]): The number of points which each SA
            module samples.
        radii (tuple[float]): Sampling radii of each SA module.
        num_samples (tuple[int]): The number of samples for ball
            query in each SA module.
        sa_channels (tuple[tuple[int]]): Out channels of each mlp in SA module.
        aggregation_channels (tuple[int]): Out channels of aggregation
            multi-scale grouping features.
        fps_mods (tuple[int]): Mod of FPS for each SA module.
        fps_sample_range_lists (tuple[tuple[int]]): The number of sampling
            points which each SA module samples.
        dilated_group (tuple[bool]): Whether to use dilated ball query for
        out_indices (Sequence[int]): Output from which stages.
        norm_cfg (dict): Config of normalization layer.
        sa_cfg (dict): Config of set abstraction module, which may contain
            the following keys and values:

            - pool_mod (str): Pool method ('max' or 'avg') for SA modules.
            - use_xyz (bool): Whether to use xyz as a part of features.
            - normalize_xyz (bool): Whether to normalize xyz with radii in
              each SA module.
    """

    def __init__(self,
                 in_channels,
                 num_points=(2048, 1024, 512, 256),
                 radii=((0.2, 0.4, 0.8), (0.4, 0.8, 1.6), (1.6, 3.2, 4.8)),
                 num_samples=((32, 32, 64), (32, 32, 64), (32, 32, 32)),
                 sa_channels=(((16, 16, 32), (16, 16, 32), (32, 32, 64)),
                              ((64, 64, 128), (64, 64, 128), (64, 96, 128)),
                              ((128, 128, 256), (128, 192, 256), (128, 256,
                                                                  256))),
                 aggregation_channels=(64, 128, 256),  # 每个SA输出的Point的通道数（特征长度）
                 fps_mods=(('D-FPS'), ('FS'), ('F-FPS', 'D-FPS')),
                 fps_sample_range_lists=((-1), (-1), (512, -1)),
                 dilated_group=(True, True, True),
                 out_indices=(2,),
                 norm_cfg=dict(type='BN2d'),
                 sa_cfg=dict(
                     type='PointSAModuleMSG',
                     pool_mod='max',
                     use_xyz=True,
                     normalize_xyz=False),
                 init_cfg=None,
                 img_channels: List[int] = None):
        super().__init__(init_cfg=init_cfg)
        self.num_sa = len(sa_channels)
        self.out_indices = out_indices
        assert max(out_indices) < self.num_sa
        assert len(num_points) == len(radii) == len(num_samples) == len(
            sa_channels) == len(aggregation_channels)

        self.SA_modules = nn.ModuleList()
        self.Img_Block = nn.ModuleList()  # img backbone
        self.Fusion_Conv = nn.ModuleList()  # fusion_layer
        self.aggregation_mlps = nn.ModuleList()
        sa_in_channel = in_channels - 3  # number of channels without xyz
        skip_channel_list = [sa_in_channel]

        for sa_index in range(self.num_sa):
            cur_sa_mlps = list(sa_channels[sa_index])
            sa_out_channel = 0
            for radius_index in range(len(radii[sa_index])):
                cur_sa_mlps[radius_index] = [sa_in_channel] + list(
                    cur_sa_mlps[radius_index])
                sa_out_channel += cur_sa_mlps[radius_index][-1]

            if isinstance(fps_mods[sa_index], tuple):
                cur_fps_mod = list(fps_mods[sa_index])
            else:
                cur_fps_mod = list([fps_mods[sa_index]])

            if isinstance(fps_sample_range_lists[sa_index], tuple):
                cur_fps_sample_range_list = list(
                    fps_sample_range_lists[sa_index])
            else:
                cur_fps_sample_range_list = list(
                    [fps_sample_range_lists[sa_index]])

            self.SA_modules.append(
                build_sa_module(
                    num_point=num_points[sa_index],
                    radii=radii[sa_index],
                    sample_nums=num_samples[sa_index],
                    mlp_channels=cur_sa_mlps,
                    fps_mod=cur_fps_mod,
                    fps_sample_range_list=cur_fps_sample_range_list,
                    dilated_group=dilated_group[sa_index],
                    norm_cfg=norm_cfg,
                    cfg=sa_cfg,
                    bias=True))
            skip_channel_list.append(sa_out_channel)

            cur_aggregation_channel = aggregation_channels[sa_index]
            if cur_aggregation_channel is None:
                self.aggregation_mlps.append(None)
                sa_in_channel = sa_out_channel
            else:
                self.aggregation_mlps.append(
                    ConvModule(
                        sa_out_channel,
                        cur_aggregation_channel,
                        conv_cfg=dict(type='Conv1d'),
                        norm_cfg=dict(type='BN1d'),
                        kernel_size=1,
                        bias=True))
                sa_in_channel = cur_aggregation_channel

        # Img backbone and fusion layer
        for i in range(len(img_channels) - 1):  # [3, 64, 128, 256, 512]
            self.Img_Block.append(
                BasicBlock(img_channels[i], img_channels[i + 1], stride=1))
            self.Fusion_Conv.append(
                AttenFusionConv(img_channels[i + 1], aggregation_channels[i], aggregation_channels[i]))

    @auto_fp16(apply_to=('points',))
    def forward(self,
                points,
                img=None,
                xy=None):
        """Forward pass.

        Args:
            points (torch.Tensor): point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).
            img: 图片(B, W, H)
            xy: 点云在图片上xy的坐标(B, N, 2)

        Returns:
            dict[str, torch.Tensor]: Outputs of the last SA module.

                - sa_xyz (torch.Tensor): The coordinates of sa features.
                - sa_features (torch.Tensor): The features from the
                    last Set Aggregation Layers.
                - sa_indices (torch.Tensor): Indices of the \
                    input points.
        """
        xyz, features = self._split_point_feats(points)

        batch, num_points = xyz.shape[:2]
        indices = xyz.new_tensor(range(num_points)).unsqueeze(0).repeat(
            batch, 1).long()

        sa_xyz = [xyz]
        sa_features = [features]
        sa_indices = [indices]

        out_sa_xyz = [xyz]
        out_sa_features = [features]
        out_sa_indices = [indices]

        """
        normalize xy to [-1,1]。为什么？
        答：后面的grid_sample()需要将xy先归一化到(-1, 1)
        W为图片宽度
        x / W 取值范围(0, 1)
        x / W * 2 取值范围(0, 2)
        x / W * 2 -1 取值范围(-1, 1)
        y同理
        xy: (B, N, 2)
        """
        size_range = [1280.0, 384.0]
        xy[:, :, 0] = xy[:, :, 0] / (size_range[0] - 1.0) * 2.0 - 1.0
        xy[:, :, 1] = xy[:, :, 1] / (size_range[1] - 1.0) * 2.0 - 1.0
        # = xy / (size_range - 1.) * 2 - 1.
        l_xy_cor = [xy]
        imgs = [img.float()]

        for i in range(self.num_sa):
            # 获取点云特征
            cur_xyz, cur_features, cur_indices = self.SA_modules[i](
                sa_xyz[i], sa_features[i])
            if self.aggregation_mlps[i] is not None:
                cur_features = self.aggregation_mlps[i](cur_features)

            cur_indices_copy = cur_indices.long().unsqueeze(-1).repeat(1, 1, 2)
            li_xy_cor = torch.gather(l_xy_cor[i], dim=1, index=cur_indices_copy)  # 采样点的xy坐标。(B, M, 2)
            # 获取图片特征
            image_feature_map = self.Img_Block[i](imgs[i])  # 图像下采样
            img_gather_feature = feature_gather(image_feature_map, li_xy_cor)  # 获取点在图片上的特征。li_xy_cor为点的坐标
            # 点云、图像融合
            cur_features = self.Fusion_Conv[i](cur_features, img_gather_feature)
            imgs.append(image_feature_map)
            l_xy_cor.append(li_xy_cor)

            sa_xyz.append(cur_xyz)
            sa_features.append(cur_features)
            sa_indices.append(
                torch.gather(sa_indices[-1], 1, cur_indices.long()))
            if i in self.out_indices:
                out_sa_xyz.append(sa_xyz[-1])
                out_sa_features.append(sa_features[-1])
                out_sa_indices.append(sa_indices[-1])

        return dict(
            sa_xyz=out_sa_xyz,
            sa_features=out_sa_features,
            sa_indices=out_sa_indices)
