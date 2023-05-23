import numpy as np
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.points import get_points_type
from mmdet3d.datasets.transforms.loading import LoadPointsFromFile
# from pyntcloud import PyntCloud


@TRANSFORMS.register_module()
class MegviiLoadPointsFromFile(LoadPointsFromFile):

    def _load_pcd_points(self, pts_filename: str) -> np.ndarray:
        """读取 pcd 文件， 得到 np.ndarray(N, 4)
        """
        with open(pts_filename, 'rb') as f:
            data = f.read()
            data_binary = data[data.find(b"DATA binary") + 12:]
            points = np.frombuffer(data_binary, dtype=np.float32).reshape(-1, 3)
            points = points.astype(np.float32)
        return points

    def transform(self, results: dict) -> dict:
        """
            生成 LiDARPoints 格式
        """
        pts_file_path = results['lidar_path']
        points = self._load_pcd_points(pts_file_path)  # (N, 4)

        points = points[:, self.use_dim]

        points_class = get_points_type(self.coord_type)
        points = points_class(points, points_dim=points.shape[-1])
        results['points'] = points

        return results
