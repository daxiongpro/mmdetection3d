import torch
import numpy as np
from mmcv.runner import auto_fp16, BaseModule
from torch import nn as nn
import torch.nn.functional as F
from mmdet.models import BACKBONES

from mmdet3d.ops import furthest_point_sample


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]

    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def knn_point(k, xyz, new_xyz):
    """
    K nearest neighborhood.

    Input:
        k: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]

    Output:
        group_idx: grouped points index, [B, S, k]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, k, dim=-1, largest=False, sorted=False)
    return group_idx


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]

    Output:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def sample_and_knn_group(s, k, coords, features):
    """
    Sampling by FPS and grouping by KNN.

    Input:
        s[int]: number of points to be sampled by FPS
        k[int]: number of points to be grouped into a neighbor by KNN
        coords[tensor]: input points coordinates data with size of [B, N, 3]
        features[tensor]: input points features data with size of [B, N, D]

    Returns:
        new_coords[tensor]: sampled and grouped points coordinates by FPS with size of [B, s, k, 3]
        new_features[tensor]: sampled and grouped points features by FPS with size of [B, s, k, 2D]
    """
    batch_size = coords.shape[0]
    coords = coords.contiguous()

    # FPS sampling
    fps_idx = furthest_point_sample(coords, s).long()  # [B, s]
    new_coords = index_points(coords, fps_idx)  # [B, s, 3]
    new_features = index_points(features, fps_idx)  # [B, s, D]

    # K-nn grouping
    idx = knn_point(k, coords, new_coords)  # [B, s, k]
    grouped_features = index_points(features, idx)  # [B, s, k, D]

    # Matrix sub
    grouped_features_norm = grouped_features - new_features.view(batch_size, s, 1, -1)  # [B, s, k, D]

    # Concat
    aggregated_features = torch.cat([grouped_features_norm, new_features.view(batch_size, s, 1, -1).repeat(1, 1, k, 1)],
                                    dim=-1)  # [B, s, k, 2D]

    return new_coords, aggregated_features  # [B, s, 3], [B, s, k, 2D]


class SG(nn.Module):
    """
    SG(sampling and grouping) module.
    """

    def __init__(self, s, in_channels, out_channels):
        super(SG, self).__init__()

        self.s = s

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x, coords):
        """
        Input:
            x: features with size of [B, in_channels//2, N]
            coords: coordinates data with size of [B, N, 3]
        """
        x = x.permute(0, 2, 1)  # (B, N, in_channels//2)
        new_xyz, new_feature = sample_and_knn_group(s=self.s, k=32, coords=coords,
                                                    features=x)  # [B, s, 3], [B, s, 32, in_channels]
        b, s, k, d = new_feature.size()
        new_feature = new_feature.permute(0, 1, 3, 2)
        new_feature = new_feature.reshape(-1, d, k)  # [Bxs, in_channels, 32]
        batch_size = new_feature.size(0)
        new_feature = F.relu(self.bn1(self.conv1(new_feature)))  # [Bxs, in_channels, 32]
        new_feature = F.relu(self.bn2(self.conv2(new_feature)))  # [Bxs, in_channels, 32]
        new_feature = F.adaptive_max_pool1d(new_feature, 1).view(batch_size, -1)  # [Bxs, in_channels]
        new_feature = new_feature.reshape(b, s, -1).permute(0, 2, 1)  # [B, in_channels, s]
        return new_xyz, new_feature


class NeighborEmbedding(nn.Module):
    def __init__(self, samples=[4096, 512, 256]):
        super(NeighborEmbedding, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        self.sg1 = SG(s=samples[0], in_channels=128, out_channels=128)
        self.sg2 = SG(s=samples[1], in_channels=256, out_channels=256)
        self.sg3 = SG(s=samples[2], in_channels=512, out_channels=256)

    def forward(self, x):
        """
        Input:
            x: [B, 3, N]
        """
        xyz = x.permute(0, 2, 1)  # [B, N ,3]

        features = F.relu(self.bn1(self.conv1(x)))  # [B, 64, N]
        features = F.relu(self.bn2(self.conv2(features)))  # [B, 64, N]

        xyz1, features1 = self.sg1(features, xyz)  # [B, 128, 4096]
        xyz2, features2 = self.sg2(features1, xyz1)  # [B, 256, 512]
        xyz3, features3 = self.sg3(features2, xyz2)  # [B, 256, 256]

        return features3


class OA(nn.Module):
    """
    Offset-Attention Module.
    """

    def __init__(self, channels):
        super(OA, self).__init__()
        # channels // 4 为每个Q和K的通道数
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)

        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)  # change dim to -2 and change the sum(dim=1, keepdims=True) to dim=2

    def forward(self, x):
        """
        Input:
            x: [B, de, N]

        Output:
            x: [B, de, N]
        """
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)

        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))  # here

        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r

        return x


@BACKBONES.register_module()
class PCT(BaseModule):

    def __init__(self, samples=[4096, 1024, 512]):
        super().__init__()

        self.neighbor_embedding = NeighborEmbedding(samples)

        self.oa1 = OA(256)
        self.oa2 = OA(256)
        self.oa3 = OA(256)
        self.oa4 = OA(256)

        self.linear = nn.Sequential(
            nn.Conv1d(1280, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )

    @auto_fp16(apply_to=('points',))
    def forward(self, points):
        # points: B, 4, N = B, 4, 16384
        points = points[:, :3, :]  # 只取xyz，不要强度值
        xyz = self.neighbor_embedding(points)  # B C N

        x1 = self.oa1(xyz)
        x2 = self.oa2(x1)
        x3 = self.oa3(x2)
        x4 = self.oa4(x3)

        x = torch.cat([xyz, x1, x2, x3, x4], dim=1)

        features = self.linear(x)  # b c n =  4 256 512

        return dict(
            sa_xyz=points.permute(0, 2, 1).contiguous(),  # b n 3
            sa_features=features  # b c n
        )


if __name__ == '__main__':
    pct = PCT().cuda()

    # xyz = np.fromfile('tests/data/kitti/training/velodyne/000000.bin', dtype=np.float32)
    xyz = torch.rand(4, 3, 16384)
    xyz = xyz.cuda()  # (B, 3, N)
    out = pct(xyz)
