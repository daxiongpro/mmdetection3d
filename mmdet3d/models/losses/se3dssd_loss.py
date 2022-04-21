import torch
import torch.nn as nn

from ..builder import LOSSES
from mmdet.models.losses.utils import weighted_loss

@weighted_loss
def my_loss(pred, target):
    pass


@LOSSES.register_module()
class CELoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(CELoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        loss_bbox = self.loss_weight * my_loss(pred, target)
        return loss_bbox


@LOSSES.register_module()
class ODIouLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(ODIouLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        pass
