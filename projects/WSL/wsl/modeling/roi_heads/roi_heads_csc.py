# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import inspect
import logging
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler, convert_boxes_to_pooler_format
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.keypoint_head import build_keypoint_head
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.structures import Boxes, ImageList, Instances

from wsl.layers import CSC
from wsl.modeling.roi_heads.fast_rcnn import WSDDNOutputLayers
from wsl.modeling.roi_heads.roi_heads import (
    ROIHeads,
    get_image_level_gt,
    select_foreground_proposals,
    select_proposals_with_visible_keypoints,
)
from wsl.modeling.roi_heads.third_party.cpg_stats import Statistic

logger = logging.getLogger(__name__)


@ROI_HEADS_REGISTRY.register()
class CSCROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[ROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[ROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        csc_max_iter: int = 35000,
        output_dir: str = None,
        **kwargs
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask head.
                None if not using mask head.
            mask_pooler (ROIPooler): pooler to extra region features for mask head
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head
        self.keypoint_on = keypoint_in_features is not None
        if self.keypoint_on:
            self.keypoint_in_features = keypoint_in_features
            self.keypoint_pooler = keypoint_pooler
            self.keypoint_head = keypoint_head

        self.train_on_pred_boxes = train_on_pred_boxes

        # CSC
        self.output_dir = output_dir
        self.csc_max_iter = csc_max_iter
        self.iter = 0
        self.tau = 0.7
        self.fg_threshold = 0.1
        self.bg_threshold = 0.005
        self.csc = CSC(
            tau=self.tau,
            debug_info=False,
            fg_threshold=self.fg_threshold,
            mass_threshold=0.2,
            density_threshold=0.0,
            area_sqrt=True,
            context_scale=1.8,
        )

        self.csc_stats = Statistic(csc_max_iter, self.tau, 4, self.num_classes, output_dir, "")

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        if inspect.ismethod(cls._init_mask_head):
            ret.update(cls._init_mask_head(cfg, input_shape))
        if inspect.ismethod(cls._init_keypoint_head):
            ret.update(cls._init_keypoint_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        box_predictor = WSDDNOutputLayers(cfg, box_head.output_shape)

        output_dir = cfg.OUTPUT_DIR
        csc_max_iter = cfg.WSL.CSC_MAX_ITER

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
            "csc_max_iter": csc_max_iter,
            "output_dir": output_dir,
        }

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"mask_in_features": in_features}
        ret["mask_pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        ret["mask_head"] = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )
        return ret

    @classmethod
    def _init_keypoint_head(cls, cfg, input_shape):
        if not cfg.MODEL.KEYPOINT_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)  # noqa
        sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"keypoint_in_features": in_features}
        ret["keypoint_pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        ret["keypoint_head"] = build_keypoint_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )
        return ret

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        self.gt_classes_img, self.gt_classes_img_int, self.gt_classes_img_oh = get_image_level_gt(
            targets, self.num_classes
        )

        # del images
        self.images = images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))

            self.iter = self.iter + 1

            return proposals, losses
        else:
            pred_instances, all_scores, all_boxes = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances, _, _ = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}, all_scores, all_boxes

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances, [], []

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])

        objectness_logits = torch.cat([x.objectness_logits + 1 for x in proposals], dim=0)
        box_features = box_features * objectness_logits.view(-1, 1, 1, 1)

        # torch.cuda.empty_cache()

        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        pred_class_logits, _ = predictions
        del box_features

        cpgs = self._forward_cpg(pred_class_logits, proposals)
        W_pos, W_neg, PL, NL = self._forward_csc(cpgs, pred_class_logits, proposals)
        self._save_mask(cpgs, "cpgs", "cpg")

        if self.training:
            losses = self.box_predictor.losses_csc(
                predictions, proposals, W_pos, W_neg, PL, NL, self.csc_stats
            )
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _, all_scores, all_boxes = self.box_predictor.inference(
                predictions, proposals
            )
            return pred_instances, all_scores, all_boxes

    def _forward_mask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        features = [features[f] for f in self.mask_in_features]

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            return self.mask_head(mask_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            return self.mask_head(mask_features, instances)

    def _forward_keypoint(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        features = [features[f] for f in self.keypoint_in_features]

        if self.training:
            # The loss is defined on positive proposals with >=1 visible keypoints.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals = select_proposals_with_visible_keypoints(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            keypoint_features = self.keypoint_pooler(features, proposal_boxes)
            return self.keypoint_head(keypoint_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            keypoint_features = self.keypoint_pooler(features, pred_boxes)
            return self.keypoint_head(keypoint_features, instances)

    @torch.no_grad()
    def _forward_cpg(self, pred_class_logits, proposals):
        if not self.training:
            return None

        if self.iter > self.csc_max_iter:
            return None

        pred_class_img_logits = torch.sum(pred_class_logits, dim=0, keepdim=True)
        # pred_class_img_logits = torch.clamp(pred_class_img_logits, min=1e-6, max=1.0 - 1e-6)

        image_sizes = self.images.image_sizes[0]
        cpgs = torch.zeros(
            (1, self.num_classes, image_sizes[0], image_sizes[1]),
            dtype=pred_class_img_logits.dtype,
            device=pred_class_img_logits.device,
        )
        for c in range(self.num_classes):
            if self.gt_classes_img_oh[0, c] < 0.5:
                continue
            if pred_class_img_logits[0, c] < self.tau:
                continue

            grad_outputs = torch.zeros(
                pred_class_logits.size(),
                dtype=pred_class_logits.dtype,
                device=pred_class_logits.device,
            )
            grad_outputs[:, c] = 1.0
            (cpg,) = torch.autograd.grad(  # grad_outputs[0, c] = self.pred_class_img_logits[0, c]
                outputs=pred_class_logits,
                inputs=self.images.tensor,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False,
            )
            cpg.abs_()
            cpg, _ = torch.max(cpg, dim=1)

            # cpg_scale_op
            max_value = torch.max(cpg)
            cpg = cpg / max_value

            cpgs[0, c, :, :] = cpg[0, :, :]
            del cpg
            del grad_outputs
            torch.cuda.empty_cache()

        self.images.tensor.requires_grad = False
        self.images.tensor.detach()

        return cpgs

    @torch.no_grad()
    def _forward_csc(self, masks, pred_class_logits, proposals):
        if not self.training:
            return None, None, None, None

        if self.iter > self.csc_max_iter:
            PL = self.gt_classes_img_oh
            NL = torch.zeros(
                self.gt_classes_img_oh.size(),
                dtype=self.gt_classes_img_oh.dtype,
                device=self.gt_classes_img_oh.device,
            )
            W_pos = torch.ones(
                pred_class_logits.size(),
                dtype=pred_class_logits.dtype,
                device=pred_class_logits.device,
            )
            W_neg = torch.zeros(
                pred_class_logits.size(),
                dtype=pred_class_logits.dtype,
                device=pred_class_logits.device,
            )
            return W_pos, W_neg, PL, NL

        pred_class_img_logits = torch.sum(pred_class_logits, dim=0, keepdim=True)

        pooler_fmt_boxes = convert_boxes_to_pooler_format([x.proposal_boxes for x in proposals])

        W, PL, NL = self.csc(masks, self.gt_classes_img_oh, pred_class_img_logits, pooler_fmt_boxes)

        W_pos = torch.clamp(W, min=0.0)
        W_neg = torch.clamp(W, max=0.0)

        W_pos.abs_()
        W_neg.abs_()

        return W_pos, W_neg, PL, NL

    @torch.no_grad()
    def _save_mask(self, masks, prefix, suffix):
        if masks is None:
            return
        if self.iter % 1280 > 0:
            return

        output_dir = os.path.join(self.output_dir, prefix)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        id_str = "iter" + str(self.iter) + "_gpu" + str(masks.device.index)

        for c in range(self.num_classes):
            if self.gt_classes_img_oh[0, c] < 0.5:
                continue
            # if self.pred_class_img_logits[0, c] < self.tau:
            # continue

            mask = masks[0, c, ...].clone().detach().cpu().numpy()
            max_value = np.max(mask)
            if max_value > 0:
                max_value = max_value * 0.1
                mask = np.clip(mask, 0, max_value)
                mask = mask / max_value * 255
            mask = mask.astype(np.uint8)
            im_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

            save_path = os.path.join(output_dir, id_str + "_c" + str(c) + "_" + suffix + ".png")
            cv2.imwrite(save_path, im_color)

        im = self.images.tensor[0, ...].clone().detach().cpu().numpy()
        channel_swap = (1, 2, 0)
        im = im.transpose(channel_swap)
        pixel_means = [103.939, 116.779, 123.68]
        im += pixel_means
        im = im.astype(np.uint8)
        save_path = os.path.join(output_dir, id_str + ".png")
        cv2.imwrite(save_path, im)
