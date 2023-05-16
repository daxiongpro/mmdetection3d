# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp
from tools.dataset_converters import nuscenes_converter as nuscenes_converter
from projects.megvii_dataset.data_converters import megvii_converter as megvii_converter
from tools.dataset_converters.create_gt_database import (
    GTDatabaseCreater, create_groundtruth_database)
from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos


def megvii_data_prep(root_path,
                     info_prefix,
                     dataset_name,
                     out_dir):

    megvii_converter.create_megvii_infos(root_path, info_prefix)

    # info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    # info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    # update_pkl_infos('megvii', out_dir=out_dir, pkl_path=info_train_path)
    # update_pkl_infos('megvii', out_dir=out_dir, pkl_path=info_val_path)
    # create_groundtruth_database(dataset_name, root_path, info_prefix,
    #                             f'{info_prefix}_infos_train.pkl')


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--with-plane',
    action='store_true',
    help='Whether to use plane information for kitti.')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    from mmdet3d.utils import register_all_modules
    register_all_modules()

    if args.dataset == 'megvii':
        megvii_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            dataset_name='MegviiDataset',
            out_dir=args.out_dir)
    else:
        raise NotImplementedError(f'Don\'t support {args.dataset} dataset.')
