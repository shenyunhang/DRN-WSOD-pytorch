# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import numpy as np
import os

from detectron2.data.datasets.register_coco import register_coco_instances

# fmt: off
CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
# fmt: on

# ==== Predefined datasets and splits for Flickr ==========

_PREDEFINED_SPLITS_WEB = {}
_PREDEFINED_SPLITS_WEB["flickr"] = {
    "flickr_voc": ("flickr_voc/images", "flickr_voc/images.json"),
    "flickr_coco": ("flickr_coco/images", "flickr_coco/images.json"),
}


def register_all_web(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_WEB.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(key),
                # TODO add COCO class_names
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


def register_all_voc_sbd(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_VOC_SBD.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_builtin_metadata(key),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits""" ""
    return "".join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap


def _get_builtin_metadata(dataset_name):
    # thing_ids = [i for i, k in enumerate(CLASS_NAMES)]
    # thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = CLASS_NAMES
    thing_colors = labelcolormap(len(CLASS_NAMES))
    ret = {
        # "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


# Register them all under "./datasets"
_root = os.getenv("wsl_DATASETS", "datasets")
register_all_web(_root)
