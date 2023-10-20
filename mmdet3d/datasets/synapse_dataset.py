import copy
import glob
import os
import os.path as osp
import mmengine
from typing import Callable, List, Optional, Sequence, Union

from tifffile import tifffile

from mmdet3d.registry import DATASETS
from mmengine.dataset import BaseDataset
import numpy as np


def poly2obb_np(ploy):
    pass


@DATASETS.register_module()
class SynapseDataset(BaseDataset):
    METAINFO = {
        'classes': ('synapse',)
    }

    def __init__(self,
                 data_root: Optional[str] = None,
                 anno_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(
                     annotation='',
                     vol='',
                 ),
                 pipeline: List[Union[dict, Callable]] = [],
                 ignore_index: Optional[int] = None,
                 test_mode: bool = False,
                 load_eval_anns: bool = True,
                 backend_args: Optional[dict] = None,
                 **kwargs) -> None:
        self.anno_file = anno_file
        self.backend_args = backend_args
        self.load_eval_anns = load_eval_anns

        self.ignore_index = len(self.METAINFO['classes']) if \
            ignore_index is None else ignore_index
        self.test_mode = test_mode
        super(SynapseDataset, self).__init__(
            data_root=data_root,
            ann_file=anno_file,
            pipeline=pipeline,
            metainfo=metainfo,
            **kwargs)

    def load_data_list(self):
        data_infos = []
        ann_files = glob.glob(self.anno_file + '/*.txt')

        if 1:
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.tif'
                data_info['filename'] = img_name
                vol = tifffile.imread(os.path.join(self.data_root, f'volume/{img_name}'))
                data_info['volume'] = vol.astype(np.float32)
                data_info['ann_info'] = {}
                gt_bboxes = []
                gt_labels = []

                with open(ann_file) as f:
                    s = f.readlines()
                    for si in s:
                        bbox_info = si.split()
                        poly = np.array(bbox_info[:9], dtype=np.float32)
                        try:
                            z, y, x, d, h, w, theta_z, theta_y, theta_x = poly
                        except:  # noqa: E722
                            continue
                        # todo only one label supported, change here
                        gt_bboxes.append([x, y, z, w, h, d, theta_x, theta_y, theta_z])
                        gt_labels.append(1)
                if gt_bboxes:
                    data_info['ann_info']['gt_bboxes_3d'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann_info']['gt_labels_3d'] = np.array(
                        gt_labels, dtype=np.int32)
                data_infos.append(data_info)
        return data_infos


    # def parse_data_info(self, info: dict) -> dict:
    #     print('2222222222222222222222222222222222222222222222222')
    #     if not self.test_mode:
    #         # used in training
    #         info['ann_info'] = self.parse_ann_info(info)
    #     if self.test_mode and self.load_eval_anns:
    #         info['eval_ann_info'] = self.parse_ann_info(info)
    #
    #     return info

    # def parse_ann_info(self, info: dict):
    #     """Process the `instances` in data info to `ann_info`.
    #
    #     In `Custom3DDataset`, we simply concatenate all the field
    #     in `instances` to `np.ndarray`, you can do the specific
    #     process in subclass. You have to convert `gt_bboxes_3d`
    #     to different coordinates according to the task.
    #
    #     Args:
    #         info (dict): Info dict.
    #
    #     Returns:
    #         dict or None: Processed `ann_info`.
    #     """
    #     # add s or gt prefix for most keys after concat
    #     # we only process 3d annotations here, the corresponding
    #     # 2d annotation process is in the `LoadAnnotations3D`
    #     # in `transforms`
    #     ann_info = super().parse_ann_info(info)
    #     if ann_info is None:
    #         ann_info = dict()
    #         # empty instance
    #         ann_info['gt_bboxes_3d'] = np.zeros((0, 9), dtype=np.float32)
    #         ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)
    #
    #     data_infos = []
    #     ann_files = glob.glob(info['anno_file'] + '/*.txt')
    #     if not ann_files:  # test phase
    #         ann_files = glob.glob(info['anno_file'] + '/*.png')
    #         for ann_file in ann_files:
    #             data_info = {}
    #             img_id = osp.split(ann_file)[1][:-4]
    #             img_name = img_id + '.png'
    #             data_info['filename'] = img_name
    #             data_info['ann'] = {}
    #             data_info['ann']['bboxes'] = []
    #             data_info['ann']['labels'] = []
    #
    #     else:
    #         for ann_file in ann_files:
    #             data_info = {}
    #             img_id = osp.split(ann_file)[1][:-4]
    #             img_name = img_id + '.png'
    #             data_info['filename'] = img_name
    #             data_info['ann'] = {}
    #             gt_bboxes = []
    #             gt_labels = []
    #
    #             if os.path.getsize(ann_file) == 0 and self.filter_empty_gt:
    #                 continue
    #
    #             with open(ann_file) as f:
    #                 s = f.readlines()
    #                 for si in s:
    #                     bbox_info = si.split()
    #                     poly = np.array(bbox_info[:8], dtype=np.float32)
    #                     try:
    #                         x, y, z, w, h, d, theta_x, theta_y, theta_z = poly
    #                     except:  # noqa: E722
    #                         continue
    #                     # todo only one label supported, change here
    #                     cls_name = bbox_info[9]
    #                     gt_bboxes.append([x, y, z, w, h, d, theta_x, theta_y, theta_z])
    #                     gt_labels.append('synapse')
    #             if gt_bboxes:
    #                 data_info['ann_info']['gt_bboxes_3d'] = np.array(
    #                     gt_bboxes, dtype=np.float32)
    #                 data_info['ann_info']['gt_labels_3d'] = np.array(
    #                     gt_labels, dtype=np.int64)
    #             data_infos.append(data_info)
    #     return {}

    # def prepare_data(self, index: int) -> Union[dict, None]:
    #     """Data preparation for both training and testing stage.
    #
    #     Called by `__getitem__`  of dataset.
    #
    #     Args:
    #         index (int): Index for accessing the target data.
    #
    #     Returns:
    #         dict or None: Data dict of the corresponding index.
    #     """
    #     ori_input_dict = self.get_data_info(index)
    #
    #     # deepcopy here to avoid inplace modification in pipeline.
    #     input_dict = copy.deepcopy(ori_input_dict)
    #
    #     # box_type_3d (str): 3D box type.
    #     input_dict['box_type_3d'] = self.box_type_3d
    #     # box_mode_3d (str): 3D box mode.
    #     input_dict['box_mode_3d'] = self.box_mode_3d
    #
    #     # pre-pipline return None to random another in `__getitem__`
    #     if not self.test_mode and self.filter_empty_gt:
    #         if len(input_dict['ann_info']['gt_labels_3d']) == 0:
    #             return None
    #
    #     example = self.pipeline(input_dict)
    #
    #     if not self.test_mode and self.filter_empty_gt:
    #         # after pipeline drop the example with empty annotations
    #         # return None to random another in `__getitem__`
    #         if example is None or len(
    #                 example['data_samples'].gt_instances_3d.labels_3d) == 0:
    #             return None
    #
    #     return example
