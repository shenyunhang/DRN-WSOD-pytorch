from __future__ import absolute_import
import numpy as np
from sklearn.cluster import KMeans

from detectron2.structures import Boxes, pairwise_iou

# import utils.boxes as box_utils
# from core.config import cfg

cfg_TRAIN_NUM_KMEANS_CLUSTER = 3
cfg_RNG_SEED = 3
cfg_TRAIN_GRAPH_IOU_THRESHOLD = 0.4
cfg_TRAIN_MAX_PC_NUM = 5
cfg_TRAIN_FG_THRESH = 0.5
cfg_TRAIN_BG_THRESH = 0.1


try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def PCL(boxes, cls_prob, im_labels, cls_prob_new):

    cls_prob = cls_prob.data.cpu().numpy()
    cls_prob_new = cls_prob_new.data.cpu().numpy()
    # print(cls_prob.shape, cls_prob_new.shape, im_labels.shape)
    if cls_prob.shape[1] != im_labels.shape[1]:
        cls_prob = cls_prob[:, 1:]
    eps = 1e-9
    cls_prob[cls_prob < eps] = eps
    cls_prob[cls_prob > 1 - eps] = 1 - eps
    cls_prob_new[cls_prob_new < eps] = eps
    cls_prob_new[cls_prob_new > 1 - eps] = 1 - eps

    proposals = _get_graph_centers(boxes.copy(), cls_prob.copy(), im_labels.copy())

    (
        labels,
        cls_loss_weights,
        gt_assignment,
        pc_labels,
        pc_probs,
        pc_count,
        img_cls_loss_weights,
    ) = _get_proposal_clusters(boxes.copy(), proposals, im_labels.copy(), cls_prob_new.copy())

    return {
        "labels": labels.reshape(1, -1).astype(np.float32).copy(),
        "cls_loss_weights": cls_loss_weights.reshape(1, -1).astype(np.float32).copy(),
        "gt_assignment": gt_assignment.reshape(1, -1).astype(np.float32).copy(),
        "pc_labels": pc_labels.reshape(1, -1).astype(np.float32).copy(),
        "pc_probs": pc_probs.reshape(1, -1).astype(np.float32).copy(),
        "pc_count": pc_count.reshape(1, -1).astype(np.float32).copy(),
        "img_cls_loss_weights": img_cls_loss_weights.reshape(1, -1).astype(np.float32).copy(),
        "im_labels_real": np.hstack((np.array([[1]]), im_labels)).astype(np.float32).copy(),
    }


def _get_top_ranking_propoals(probs):
    """Get top ranking proposals by k-means"""
    n_clusters = min(cfg_TRAIN_NUM_KMEANS_CLUSTER, probs.shape[0])
    kmeans = KMeans(n_clusters=n_clusters, random_state=cfg_RNG_SEED).fit(probs)
    high_score_label = np.argmax(kmeans.cluster_centers_)

    index = np.where(kmeans.labels_ == high_score_label)[0]

    if len(index) == 0:
        index = np.array([np.argmax(probs)])

    return index


def _build_graph(boxes, iou_threshold):
    """Build graph based on box IoU"""
    # overlaps = box_utils.bbox_overlaps(
    # boxes.astype(dtype=np.float32, copy=False),
    # boxes.astype(dtype=np.float32, copy=False))
    overlaps = pairwise_iou(Boxes(boxes), Boxes(boxes))
    overlaps = overlaps.data.cpu().numpy()

    return (overlaps > iou_threshold).astype(np.float32)


def _get_graph_centers(boxes, cls_prob, im_labels):
    """Get graph centers."""

    num_images, num_classes = im_labels.shape
    assert num_images == 1, "batch size shoud be equal to 1"
    im_labels_tmp = im_labels[0, :].copy()
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)
    for i in xrange(num_classes):
        if im_labels_tmp[i] == 1:
            cls_prob_tmp = cls_prob[:, i].copy()
            idxs = np.where(cls_prob_tmp >= 0)[0]
            idxs_tmp = _get_top_ranking_propoals(cls_prob_tmp[idxs].reshape(-1, 1))
            idxs = idxs[idxs_tmp]
            boxes_tmp = boxes[idxs, :].copy()
            cls_prob_tmp = cls_prob_tmp[idxs]

            graph = _build_graph(boxes_tmp, cfg_TRAIN_GRAPH_IOU_THRESHOLD)

            keep_idxs = []
            gt_scores_tmp = []
            count = cls_prob_tmp.size
            while True:
                order = np.sum(graph, axis=1).argsort()[::-1]
                tmp = order[0]
                keep_idxs.append(tmp)
                inds = np.where(graph[tmp, :] > 0)[0]
                gt_scores_tmp.append(np.max(cls_prob_tmp[inds]))

                graph[:, inds] = 0
                graph[inds, :] = 0
                count = count - len(inds)
                if count <= 5:
                    break

            gt_boxes_tmp = boxes_tmp[keep_idxs, :].copy()
            gt_scores_tmp = np.array(gt_scores_tmp).copy()

            keep_idxs_new = np.argsort(gt_scores_tmp)[
                -1 : (-1 - min(len(gt_scores_tmp), cfg_TRAIN_MAX_PC_NUM)) : -1
            ]

            gt_boxes = np.vstack((gt_boxes, gt_boxes_tmp[keep_idxs_new, :]))
            gt_scores = np.vstack((gt_scores, gt_scores_tmp[keep_idxs_new].reshape(-1, 1)))
            gt_classes = np.vstack(
                (gt_classes, (i + 1) * np.ones((len(keep_idxs_new), 1), dtype=np.int32))
            )

            # If a proposal is chosen as a cluster center,
            # we simply delete a proposal from the candidata proposal pool,
            # because we found that the results of different strategies are similar and this strategy is more efficient
            cls_prob = np.delete(cls_prob.copy(), idxs[keep_idxs][keep_idxs_new], axis=0)
            boxes = np.delete(boxes.copy(), idxs[keep_idxs][keep_idxs_new], axis=0)

    proposals = {"gt_boxes": gt_boxes, "gt_classes": gt_classes, "gt_scores": gt_scores}

    return proposals


def _get_proposal_clusters(all_rois, proposals, im_labels, cls_prob):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    num_images, num_classes = im_labels.shape
    assert num_images == 1, "batch size shoud be equal to 1"
    # overlaps: (rois x gt_boxes)
    gt_boxes = proposals["gt_boxes"]
    gt_labels = proposals["gt_classes"]
    gt_scores = proposals["gt_scores"]
    # overlaps = box_utils.bbox_overlaps(
    # all_rois.astype(dtype=np.float32, copy=False),
    # gt_boxes.astype(dtype=np.float32, copy=False))

    overlaps = pairwise_iou(Boxes(all_rois), Boxes(gt_boxes))
    overlaps = overlaps.data.cpu().numpy()

    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_labels[gt_assignment, 0]
    cls_loss_weights = gt_scores[gt_assignment, 0]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    # fg_inds = np.where(max_overlaps >= cfg_TRAIN_FG_THRESH)[0]

    # Select background RoIs as those with < FG_THRESH overlap
    bg_inds = np.where(max_overlaps < cfg_TRAIN_FG_THRESH)[0]

    ig_inds = np.where(max_overlaps < cfg_TRAIN_BG_THRESH)[0]
    cls_loss_weights[ig_inds] = 0.0

    labels[bg_inds] = 0
    gt_assignment[bg_inds] = -1

    img_cls_loss_weights = np.zeros(gt_boxes.shape[0], dtype=np.float32)
    pc_probs = np.zeros(gt_boxes.shape[0], dtype=np.float32)
    pc_labels = np.zeros(gt_boxes.shape[0], dtype=np.int32)
    pc_count = np.zeros(gt_boxes.shape[0], dtype=np.int32)

    for i in xrange(gt_boxes.shape[0]):
        po_index = np.where(gt_assignment == i)[0]
        img_cls_loss_weights[i] = np.sum(cls_loss_weights[po_index])
        pc_labels[i] = gt_labels[i, 0]
        pc_count[i] = len(po_index)
        pc_probs[i] = np.average(cls_prob[po_index, pc_labels[i]])

    return (
        labels,
        cls_loss_weights,
        gt_assignment,
        pc_labels,
        pc_probs,
        pc_count,
        img_cls_loss_weights,
    )
