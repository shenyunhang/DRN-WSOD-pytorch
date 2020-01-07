import numpy as np
import os

from detectron2.utils.events import EventWriter
from detectron2.utils.logger import setup_logger


class Statistic(EventWriter):
    def __init__(self, max_iter, tau, num_gpus, num_classes, output_dir, prefix):
        self.tau = tau
        self.LOG_PERIOD = int(1280 / num_gpus)
        self.max_iter = max_iter
        self.cur_iter = 0

        self.num_classes = num_classes

        self.ori_label = [0.0 for c in range(self.num_classes)]
        self.ori_pred = [0.0 for c in range(self.num_classes)]
        self.ori_num_roi = [0.0 for c in range(self.num_classes)]
        self.ori_acm_label = 0.0
        self.ori_acm_pred = 0.0
        self.ori_acm_num_roi = 0.0

        self.csc_label = [0.0 for c in range(self.num_classes)]
        self.csc_pred = [0.0 for c in range(self.num_classes)]
        self.csc_pred_pos = [0.0 for c in range(self.num_classes)]
        self.csc_pred_neg = [0.0 for c in range(self.num_classes)]
        self.csc_num_roi = [0.0 for c in range(self.num_classes)]
        self.csc_roi_pos = [0.0 for c in range(self.num_classes)]
        self.csc_roi_neg = [0.0 for c in range(self.num_classes)]
        self.csc_roi_zero = [0.0 for c in range(self.num_classes)]
        self.csc_acm_label = 0.0
        self.csc_acm_pred = 0.0
        self.csc_acm_pred_pos = 0.0
        self.csc_acm_pred_neg = 0.0
        self.csc_acm_num_roi = 0.0
        self.csc_acm_roi_pos = 0.0
        self.csc_acm_roi_neg = 0.0
        self.csc_acm_roi_zero = 0.0

        self.num_img = 0

        log_path = os.path.join(output_dir, prefix + "csc.txt")
        self.logger = setup_logger(output=log_path, name=prefix + "csc")
        # self.logger = logging.getLogger("detectron2")

    def UpdateIterStats(
        self, label_val, pred_val, pred_pos_val, pred_neg_val, csc_pos_val, csc_neg_val
    ):
        if True:
            csc_val = csc_pos_val - csc_neg_val

            label_val = np.squeeze(label_val)
            pred_val = np.squeeze(pred_val)
            pred_pos_val = np.squeeze(pred_pos_val)
            pred_neg_val = np.squeeze(pred_neg_val)
            csc_val = np.squeeze(csc_val)

            for c in range(self.num_classes):
                if label_val[c] <= 0.5:
                    continue

                self.ori_label[c] += 1
                self.ori_pred[c] += pred_val[c]
                self.ori_num_roi[c] += csc_val.shape[0]

                self.ori_acm_label += 1
                self.ori_acm_pred += pred_val[c]
                self.ori_acm_num_roi += csc_val.shape[0]

                if pred_val[c] < self.tau:
                    continue

                self.csc_label[c] += 1
                self.csc_pred[c] += pred_val[c]
                self.csc_pred_pos[c] += pred_pos_val[c]
                self.csc_pred_neg[c] += pred_neg_val[c]
                self.csc_num_roi[c] += csc_val.shape[0]

                self.csc_acm_label += 1
                self.csc_acm_pred += pred_val[c]
                self.csc_acm_pred_pos += pred_pos_val[c]
                self.csc_acm_pred_neg += pred_neg_val[c]
                self.csc_acm_num_roi += csc_val.shape[0]

                for r in range(csc_val.shape[0]):
                    if csc_val[r, c] > 0:
                        self.csc_roi_pos[c] += 1
                        self.csc_acm_roi_pos += 1
                    elif csc_val[r, c] < 0:
                        self.csc_roi_neg[c] += 1
                        self.csc_acm_roi_neg += 1
                    else:
                        self.csc_roi_zero[c] += 1
                        self.csc_acm_roi_zero += 1

            self.num_img += 1

    def LogIterStats(self):
        if self.cur_iter % self.LOG_PERIOD > 0 or self.cur_iter > self.max_iter:
            self.cur_iter += 1
            return

        info_all = ""
        info_line = "----------------------------------------------"
        info_all += info_line + str(self.cur_iter) + info_line + "\n"
        info_all += (
            "#class\tpred\t#roi\t#class\tpred\tpos\tneg\t#roi\t#pos\t%\t#neg\t%\t#zero\t%\tclass"
            + "\n"
        )

        for c in range(self.num_classes):

            if self.ori_label[c] > 0:
                info_ori = "{}\t{:.5f}\t{}\t".format(
                    int(self.ori_label[c]),
                    self.ori_pred[c] / self.ori_label[c],
                    int(self.ori_num_roi[c] / self.ori_label[c]),
                )
            else:
                info_ori = "0\t0.0000\t0\t"

            self.ori_label[c] = 0.0
            self.ori_pred[c] = 0.0
            self.ori_num_roi[c] = 0.0

            if self.csc_label[c] > 0:
                info_csc = "{}\t{:.5f}\t{:.5f}\t{:.5f}\t{}\t{}\t{:.5f}\t{}\t{:.5f}\t{}\t{:.5f}\t{}".format(
                    int(self.csc_label[c]),
                    self.csc_pred[c] / self.csc_label[c],
                    self.csc_pred_pos[c] / self.csc_label[c],
                    self.csc_pred_neg[c] / self.csc_label[c],
                    int(self.csc_num_roi[c] / self.csc_label[c]),
                    int(self.csc_roi_pos[c] / self.csc_label[c]),
                    1.0 * self.csc_roi_pos[c] / self.csc_num_roi[c],
                    int(self.csc_roi_neg[c] / self.csc_label[c]),
                    1.0 * self.csc_roi_neg[c] / self.csc_num_roi[c],
                    int(self.csc_roi_zero[c] / self.csc_label[c]),
                    1.0 * self.csc_roi_zero[c] / self.csc_num_roi[c],
                    c,
                )
            else:
                info_csc = "0\t0.0000\t0.0000\t0.0000\t0\t0\t0.0000\t0\t0.0000\t0\t0.0000\t{}".format(
                    c
                )

            self.csc_label[c] = 0.0
            self.csc_pred[c] = 0.0
            self.csc_pred_pos[c] = 0.0
            self.csc_pred_neg[c] = 0.0
            self.csc_num_roi[c] = 0.0
            self.csc_roi_pos[c] = 0.0
            self.csc_roi_neg[c] = 0.0
            self.csc_roi_zero[c] = 0.0

            # self.logger.info(info_ori + info_csc)
            info_all += info_ori + info_csc + "\n"

        if self.ori_acm_label > 0 and self.csc_acm_label > 0:
            info_acm = "{}\t{:.5f}\t{}\t{}\t{:.5f}\t{:.5f}\t{:.5f}\t{}\t{}\t{:.5f}\t{}\t{:.5f}\t{}\t{:.5f}\t{}".format(
                int(self.ori_acm_label),
                self.ori_acm_pred / self.ori_acm_label,
                int(self.ori_acm_num_roi / self.ori_acm_label),
                self.csc_acm_label,
                self.csc_acm_pred / self.csc_acm_label,
                self.csc_acm_pred_pos / self.csc_acm_label,
                self.csc_acm_pred_neg / self.csc_acm_label,
                int(self.csc_acm_num_roi / self.csc_acm_label),
                int(self.csc_acm_roi_pos / self.csc_acm_label),
                1.0 * self.csc_acm_roi_pos / self.csc_acm_num_roi,
                int(self.csc_acm_roi_neg / self.csc_acm_label),
                1.0 * self.csc_acm_roi_neg / self.csc_acm_num_roi,
                int(self.csc_acm_roi_zero / self.csc_acm_label),
                1.0 * self.csc_acm_roi_zero / self.csc_acm_num_roi,
                self.num_img,
            )
        else:
            info_acm = ""

        self.ori_acm_label = 0.0
        self.ori_acm_pred = 0.0
        self.ori_acm_num_roi = 0.0

        self.csc_acm_label = 0.0
        self.csc_acm_pred = 0.0
        self.csc_acm_pred_pos = 0.0
        self.csc_acm_pred_neg = 0.0
        self.csc_acm_num_roi = 0.0
        self.csc_acm_roi_pos = 0.0
        self.csc_acm_roi_neg = 0.0
        self.csc_acm_roi_zero = 0.0

        self.num_img = 0

        # self.logger.info(info_acm)
        info_all += info_acm
        self.logger.info(info_all)

        self.cur_iter += 1
