# Copyright (c) OpenMMLab. All rights reserved.

from mmengine.evaluator import BaseMetric

from mmdet3d.registry import METRICS



@METRICS.register_module()
class NuScenesMetric(BaseMetric):
    pass