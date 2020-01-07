# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_wsl_config(cfg):
    """
    Add config for mrrpnet.
    """
    _C = cfg

    _C.MODEL.VGG = CN()

    _C.MODEL.VGG.DEPTH = 16
    _C.MODEL.VGG.OUT_FEATURES = ["plain5"]
    _C.MODEL.VGG.CONV5_DILATION = 1

    _C.WSL = CN()
    _C.WSL.VIS_TEST = False
    _C.WSL.ITER_SIZE = 1
    _C.WSL.MEAN_LOSS = True

    _C.MODEL.ROI_BOX_HEAD.DAN_DIM = [4096, 4096]

    _C.WSL.USE_OBN = True

    _C.WSL.CSC_MAX_ITER = 35000

    _C.WSL.REFINE_NUM = 3
    _C.WSL.REFINE_REG = [False, False, False]

    # List of the dataset names for testing. Must be registered in DatasetCatalog
    _C.DATASETS.VAL = ()
    # List of the pre-computed proposal files for test, which must be consistent
    # with datasets listed in DATASETS.VAL.
    _C.DATASETS.PROPOSAL_FILES_VAL = ()

    _C.MODEL.SEM_SEG_HEAD.ASSP_CONVS_DIM = [1024, 1024]
    _C.MODEL.SEM_SEG_HEAD.MASK_SOFTMAX = False
    _C.MODEL.SEM_SEG_HEAD.CONSTRAINT = False

    _C.TEST.EVAL_TRAIN = True
