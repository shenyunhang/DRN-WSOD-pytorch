from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import os
import sys
from pathlib import Path
import cv2
import scipy.io as sio
from six.moves import cPickle as pickle
from tqdm import tqdm

from detectron2.data.catalog import DatasetCatalog

import wsl.data.datasets


def convert_ss_box():
    dataset_name = sys.argv[1]
    file_in = sys.argv[2]
    file_out = sys.argv[3]

    dataset_dicts = DatasetCatalog.get(dataset_name)
    raw_data = sio.loadmat(file_in)["boxes"].ravel()
    assert raw_data.shape[0] == len(dataset_dicts)

    boxes = []
    scores = []
    ids = []
    for i in range(len(dataset_dicts)):
        if i % 1000 == 0:
            print("{}/{}".format(i + 1, len(dataset_dicts)))

        if "flickr" in dataset_name:
            index = os.path.basename(dataset_dicts[i]["file_name"])[:-4]
        elif "coco" in dataset_name:
            index = os.path.basename(dataset_dicts[i]["file_name"])[:-4]
        else:
            index = dataset_dicts[i]["image_id"]
        # selective search boxes are 1-indexed and (y1, x1, y2, x2)
        i_boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
        # i_scores = np.zeros((i_boxes.shape[0]), dtype=np.float32)
        i_scores = np.ones((i_boxes.shape[0]), dtype=np.float32)

        boxes.append(i_boxes.astype(np.int16))
        scores.append(np.squeeze(i_scores.astype(np.float32)))
        index = dataset_dicts[i]["image_id"]
        ids.append(index)

    with open(file_out, "wb") as f:
        pickle.dump(dict(boxes=boxes, scores=scores, indexes=ids), f, pickle.HIGHEST_PROTOCOL)


def convert_mcg_box():
    dataset_name = sys.argv[1]
    dir_in = sys.argv[2]
    file_out = sys.argv[3]

    dataset_dicts = DatasetCatalog.get(dataset_name)

    boxes = []
    scores = []
    ids = []
    for i in range(len(dataset_dicts)):
        if i % 1000 == 0:
            print("{}/{}".format(i + 1, len(dataset_dicts)))

        if "flickr" in dataset_name:
            index = os.path.basename(dataset_dicts[i]["file_name"])[:-4]
        elif "coco" in dataset_name:
            index = os.path.basename(dataset_dicts[i]["file_name"])[:-4]
        else:
            index = dataset_dicts[i]["image_id"]
        box_file = os.path.join(dir_in, "{}.mat".format(index))
        mat_data = sio.loadmat(box_file)
        if i == 0:
            print(mat_data.keys())
        if "flickr" in dataset_name:
            boxes_data = mat_data["bboxes"]
            scores_data = mat_data["bboxes_scores"]
        else:
            boxes_data = mat_data["boxes"]
            scores_data = mat_data["scores"]
        # selective search boxes are 1-indexed and (y1, x1, y2, x2)
        # Boxes from the MCG website are in (y1, x1, y2, x2) order
        boxes_data = boxes_data[:, (1, 0, 3, 2)] - 1
        # boxes_data_ = boxes_data.astype(np.uint16) - 1
        # boxes_data = boxes_data_[:, (1, 0, 3, 2)]

        boxes.append(boxes_data.astype(np.int16))
        scores.append(np.squeeze(scores_data.astype(np.float32)))
        index = dataset_dicts[i]["image_id"]
        ids.append(index)

    with open(file_out, "wb") as f:
        pickle.dump(dict(boxes=boxes, scores=scores, indexes=ids), f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    if "ss" in sys.argv[3].lower():
        convert_ss_box()
    elif "mcg" in sys.argv[3].lower():
        convert_mcg_box()
