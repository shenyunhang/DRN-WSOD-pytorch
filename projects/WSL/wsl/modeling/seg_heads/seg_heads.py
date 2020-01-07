# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import Dict
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec
from detectron2.modeling.meta_arch import SEM_SEG_HEADS_REGISTRY

# from wsl.layers import crf
from wsl.modeling.seg_heads.crf import dense_crf

logger = logging.getLogger(__name__)


class ASPPBranch(nn.Module):
    def __init__(self, d, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.in_features      = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        input_channels        = input_shape.channels[0]
        self.ignore_value     = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        num_classes           = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        conv_dims             = cfg.MODEL.SEM_SEG_HEAD.ASSP_CONVS_DIM
        self.common_stride    = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        norm                  = cfg.MODEL.SEM_SEG_HEAD.NORM
        self.loss_weight      = cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT
        self.mask_softmax     = cfg.MODEL.SEM_SEG_HEAD.MASK_SOFTMAX
        # fmt: on

        num_conv = len(conv_dims)
        self.layers = []
        for k in range(num_conv):
            norm_module = nn.GroupNorm(32, conv_dims) if norm == "GN" else None
            conv = Conv2d(
                input_channels if k == 0 else conv_dims[k - 1],
                conv_dims[k],
                kernel_size=3 if k == 0 else 1,
                stride=1,
                padding=d if k == 0 else 0,
                dilation=d if k == 0 else 1,
                bias=not norm,
                norm=norm_module,
                activation=F.relu_,
            )
            weight_init.c2_msra_fill(conv)
            self.add_module("conv{}".format(k + 1), conv)
            self.layers.append(conv)

            dropout = torch.nn.Dropout(p=0.5)
            self.add_module("dropout{}".format(k + 1), dropout)
            self.layers.append(dropout)

        predictor = Conv2d(
            conv_dims[-1],
            num_classes + 1 if self.mask_softmax else num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=not norm,
            norm=None,
            activation=None,
        )
        weight_init.c2_msra_fill(predictor)
        self.add_module("predictor", predictor)
        self.layers.append(predictor)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@SEM_SEG_HEADS_REGISTRY.register()
class ASPPHead(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.in_features      = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        self.ignore_value     = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        self.num_classes      = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.common_stride    = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        self.loss_weight      = cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT
        self.mask_softmax     = cfg.MODEL.SEM_SEG_HEAD.MASK_SOFTMAX
        self.constraint       = cfg.MODEL.SEM_SEG_HEAD.CONSTRAINT
        # fmt: on

        self.device = torch.device(cfg.MODEL.DEVICE)
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)

        # TODO(YH): batch size=1
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(1, num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(1, num_channels, 1, 1)
        self.img_normalizer = lambda x: (x * pixel_std + pixel_mean)

        self.max_pool = nn.MaxPool2d(3, stride=1, padding=1)

        self.dilations = [6, 12, 18, 24]

        self.scale_heads = []
        for d in self.dilations:
            branch = ASPPBranch(d, cfg, input_shape)
            self.add_module("dilation{}".format(d), branch)
            self.scale_heads.append(branch)

    def forward(self, images, features, targets=None, weights=None):
        features = features[0]
        features = self.max_pool(features)

        for i, d in enumerate(self.dilations):
            if i == 0:
                x = self.scale_heads[i](features)
            else:
                x = x + self.scale_heads[i](features)

        x_sigmoid = torch.sigmoid(x)
        if self.training:
            x = F.interpolate(x, size=targets.size()[2:], mode="bilinear", align_corners=False)
            losses = {}
            if self.mask_softmax:
                losses["loss_sem_seg"] = (
                    F.cross_entropy(x, targets, reduction="mean", ignore_index=self.ignore_value)
                    * self.loss_weight
                )
            else:
                loss_sem_seg = (
                    F.binary_cross_entropy_with_logits(x, targets, weights, reduction="none")
                    * self.loss_weight
                )
                loss_sem_seg[torch.isnan(loss_sem_seg)] = 0
                losses["loss_sem_seg"] = loss_sem_seg.sum()

            if self.constraint:
                x_crf, weights_crf = self.crf(images, x_sigmoid)
                loss_constraint = (
                    F.kl_div(torch.log(torch.sigmoid(x_sigmoid)), x_crf, reduction="none")
                    * weights_crf
                )
                loss_constraint[loss_constraint > 1000] = 0
                losses["loss_constraint"] = loss_constraint.sum()
                return [x, x_crf], losses
            return x, losses
        else:
            if self.constraint:
                # crf_size = (321, 321)
                crf_size = (513, 513)
                # crf_size = images.image_size[0]
                x_sigmoid = F.interpolate(
                    x_sigmoid, size=crf_size, mode="bilinear", align_corners=False
                )
                x_crf, _ = self.crf(images, x_sigmoid)
                return (
                    F.interpolate(
                        x_crf, size=images.image_sizes[0], mode="bilinear", align_corners=False
                    ),
                    {},
                )
            else:
                return (
                    F.interpolate(
                        x_sigmoid, size=images.image_sizes[0], mode="bilinear", align_corners=False
                    ),
                    {},
                )

    @torch.no_grad()
    def crf(self, images, x_fg):
        x_fg_max, _ = torch.max(x_fg, dim=1, keepdim=True)
        x_bg = 1.0 - x_fg_max
        # x_bg = torch.pow(x_bg, 16)
        x_bgfg = torch.cat((x_bg, x_fg), dim=1)
        # x_bgfg = F.softmax(x_bgfg, dim=1)

        x_data = F.interpolate(
            images.tensor, size=x_bgfg.size()[2:], mode="bilinear", align_corners=False
        )
        x_data = self.img_normalizer(x_data)
        # x_data = self.img_normalizer(images.tensor)

        # ========================================================================
        # densecrf
        # x_unary = -1 * torch.log(x_bgfg_softmax)
        # x_crf = crf(x_unary.cpu(), x_data.cpu())
        # ========================================================================
        # pydensecrf
        # min_prob = torch.tensor(0.0001, dtype=x_bgfg.dtype, device=x_bgfg.device)
        # x_bgfg = torch.max(x_bgfg, min_prob)
        x_crf = dense_crf(
            x_data.clone().detach().cpu().numpy().transpose((0, 2, 3, 1)),
            x_bgfg.clone().detach().cpu().numpy(),
        )
        x_crf = torch.from_numpy(x_crf).to(self.device)
        # x_crf = torch.max(x_crf, min_prob)
        # x_crf = x_crf / torch.sum(x_crf, dim=1, keepdim=True)[0]
        # ========================================================================

        x_crf_bg, x_crf_fg = torch.split(x_crf, [1, self.num_classes], dim=1)

        # 1   : pos
        # 0   : neg
        # 255 : ignore
        self.fg_threshold = 0.5
        self.bg_threshold = 0.5
        targets = torch.ones_like(x_crf_fg)
        targets[x_crf_fg < self.fg_threshold] = 255
        targets[x_crf_fg < self.bg_threshold] = 0
        # targets[pred_class_img_logits < self.tau, :, :] = 255
        # targets[self.gt_classes_img_oh == 0.5, :, :] = 255
        # targets[self.gt_classes_img_oh == 0, :, :] = 0

        pos = (targets == 1).sum(dim=[2, 3], keepdim=True).expand_as(x_crf_fg).type_as(x_crf_fg)
        neg = (targets == 0).sum(dim=[2, 3], keepdim=True).expand_as(x_crf_fg).type_as(x_crf_fg)

        weights = torch.ones_like(x_crf_fg)
        # spatial_size = cpgs.size(1) * cpgs.size(2) * cpgs.size(3)
        weights[targets == 1] = pos[targets == 1].reciprocal()
        weights[targets == 0] = neg[targets == 0].reciprocal()
        weights[targets == 255] = 0

        targets[targets == 255] = 0

        return x_crf_fg, weights
