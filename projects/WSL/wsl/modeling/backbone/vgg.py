# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
from torch import nn

from detectron2.layers import Conv2d, FrozenBatchNorm2d, ShapeSpec
from detectron2.modeling.backbone.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY

__all__ = ["PlainBlockBase", "PlainBlock", "VGG16", "build_vgg_backbone"]


class PlainBlockBase(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            in_channels (int):
            out_channels (int):
            stride (int):
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self


class PlainBlock(PlainBlockBase):
    def __init__(self, in_channels, out_channels, num_conv=3, dilation=1, stride=1, has_pool=False):
        super().__init__(in_channels, out_channels, stride)

        self.num_conv = num_conv
        self.dilation = dilation

        self.has_pool = has_pool
        self.pool_stride = stride

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1 * dilation,
            bias=True,
            groups=1,
            dilation=dilation,
            norm=None,
        )
        weight_init.c2_msra_fill(self.conv1)

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1 * dilation,
            bias=True,
            groups=1,
            dilation=dilation,
            norm=None,
        )
        weight_init.c2_msra_fill(self.conv2)

        if self.num_conv > 2:
            self.conv3 = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1 * dilation,
                bias=True,
                groups=1,
                dilation=dilation,
                norm=None,
            )
            weight_init.c2_msra_fill(self.conv3)

        if self.num_conv > 3:
            self.conv4 = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1 * dilation,
                bias=True,
                groups=1,
                dilation=dilation,
                norm=None,
            )
            weight_init.c2_msra_fill(self.conv4)

        if self.has_pool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=self.pool_stride, padding=0)

        assert num_conv < 5

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)

        x = self.conv2(x)
        x = F.relu_(x)

        if self.num_conv > 2:
            x = self.conv3(x)
            x = F.relu_(x)

        if self.num_conv > 3:
            x = self.conv4(x)
            x = F.relu_(x)

        if self.has_pool:
            x = self.pool(x)

        return x


class VGG16(Backbone):
    def __init__(self, conv5_dilation, freeze_at, num_classes=None, out_features=None):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[ResNetBlock]]): several (typically 4) stages,
                each contains multiple :class:`ResNetBlockBase`.
            num_classes (None or int): if None, will not perform classification.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
        """
        super(VGG16, self).__init__()

        self.num_classes = num_classes

        self._out_feature_strides = {}
        self._out_feature_channels = {}

        self.stages_and_names = []

        name = "plain1"
        block = PlainBlock(3, 64, num_conv=2, stride=2, has_pool=True)
        blocks = [block]
        stage = nn.Sequential(*blocks)
        self.add_module(name, stage)
        self.stages_and_names.append((stage, name))
        self._out_feature_strides[name] = 2
        self._out_feature_channels[name] = blocks[-1].out_channels
        if freeze_at >= 1:
            for block in blocks:
                block.freeze()

        name = "plain2"
        block = PlainBlock(64, 128, num_conv=2, stride=2, has_pool=True)
        blocks = [block]
        stage = nn.Sequential(*blocks)
        self.add_module(name, stage)
        self.stages_and_names.append((stage, name))
        self._out_feature_strides[name] = 4
        self._out_feature_channels[name] = blocks[-1].out_channels
        if freeze_at >= 2:
            for block in blocks:
                block.freeze()

        name = "plain3"
        block = PlainBlock(128, 256, num_conv=3, stride=2, has_pool=True)
        blocks = [block]
        stage = nn.Sequential(*blocks)
        self.add_module(name, stage)
        self.stages_and_names.append((stage, name))
        self._out_feature_strides[name] = 8
        self._out_feature_channels[name] = blocks[-1].out_channels
        if freeze_at >= 3:
            for block in blocks:
                block.freeze()

        name = "plain4"
        block = PlainBlock(
            256, 512, num_conv=3, stride=1 if conv5_dilation == 2 else 2, has_pool=True
        )
        blocks = [block]
        stage = nn.Sequential(*blocks)
        self.add_module(name, stage)
        self.stages_and_names.append((stage, name))
        self._out_feature_strides[name] = 8 if conv5_dilation == 2 else 16
        self._out_feature_channels[name] = blocks[-1].out_channels
        if freeze_at >= 4:
            for block in blocks:
                block.freeze()

        name = "plain5"
        block = PlainBlock(512, 512, num_conv=3, stride=1, dilation=conv5_dilation, has_pool=False)
        blocks = [block]
        stage = nn.Sequential(*blocks)
        self.add_module(name, stage)
        self.stages_and_names.append((stage, name))
        self._out_feature_strides[name] = 8 if conv5_dilation == 2 else 16
        self._out_feature_channels[name] = blocks[-1].out_channels
        if freeze_at >= 5:
            for block in blocks:
                block.freeze()

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

    def forward(self, x):
        outputs = {}
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x

        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_vgg_backbone(cfg, input_shape):

    # fmt: off
    depth                = cfg.MODEL.VGG.DEPTH
    conv5_dilation       = cfg.MODEL.VGG.CONV5_DILATION
    freeze_at            = cfg.MODEL.BACKBONE.FREEZE_AT
    # fmt: on

    if depth == 16:
        return VGG16(conv5_dilation, freeze_at)
