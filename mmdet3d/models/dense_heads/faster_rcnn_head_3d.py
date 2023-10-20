from mmengine.model import BaseModule
import torch.nn as nn
from mmdet3d.registry import MODELS
from typing import Dict, List, Optional, Tuple
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils.typing_utils import InstanceList
@MODELS.register_module()
class FasterRCNNHead3D(BaseModule):
    def __init__(self, num_classes, in_channels, train_cfg=None, test_cfg=None, init_cfg=None, **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fc_out_channels = in_channels
        self.fc_cls = nn.Linear(self.fc_out_channels, num_classes)
        self.fc_reg = nn.Linear(self.fc_out_channels, 6)  # for 3D bounding boxes, we have 6 regression targets
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_cfg = init_cfg

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.flatten(1)

        if self.with_fc:
            x = self.fc(x)
            if self.with_reg:
                bbox_pred = self.fc_reg(x)
            else:
                bbox_pred = None
            cls_score = self.fc_cls(x)
        else:
            if self.with_reg:
                bbox_pred = self.conv_reg(x)
            else:
                bbox_pred = None
            cls_score = self.conv_cls(x)

        return cls_score, bbox_pred

    def loss_and_predict(self,
                         feats_dict: Dict,
                         batch_data_samples: SampleList,
                         proposal_cfg: Optional[dict] = None,
                         **kwargs) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.

        Args:
            feats_dict (dict): Contains features from the first stage.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.
            proposal_cfg (ConfigDict, optional): Proposal config.

        Returns:
            tuple: the return value is a tuple contains:

            - losses: (dict[str, Tensor]): A dictionary of loss components.
            - predictions (list[:obj:`InstanceData`]): Detection
              results of each sample after the post process.
        """
        batch_gt_instances_3d = []
        batch_gt_instances_ignore = []
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            batch_gt_instances_ignore.append(
                data_sample.get('ignored_instances', None))
        raw_points = feats_dict.pop('raw_points')
        bbox_preds, cls_preds = self(feats_dict)

        loss_inputs = (bbox_preds, cls_preds,
                       raw_points) + (batch_gt_instances_3d, batch_input_metas,
                                      batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(
            raw_points,
            bbox_preds,
            cls_preds,
            batch_input_metas=batch_input_metas,
            cfg=proposal_cfg)
        feats_dict['points_cls_preds'] = cls_preds
        if predictions[0].bboxes_3d.tensor.isinf().any():
            print(predictions)
        return losses, predictions
