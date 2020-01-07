#include <math.h>

#include <ATen/TensorUtils.h>
#include "pcl_loss.h"

namespace wsl {

int pcl_loss_forward_cpu(
    at::Tensor& pcl_probs,
    at::Tensor& labels,
    at::Tensor& cls_loss_weights,
    at::Tensor& pc_labels,
    at::Tensor& pc_probs,
    at::Tensor& img_cls_loss_weights,
    at::Tensor& im_labels,
    at::Tensor& output) {
  // Grab the input tensor
  float* prob_data_flat = pcl_probs.contiguous().data_ptr<float>();
  float* labels_flat = labels.contiguous().data_ptr<float>();
  float* cls_loss_weights_flat =
      cls_loss_weights.contiguous().data_ptr<float>();
  float* pc_labels_flat = pc_labels.contiguous().data_ptr<float>();
  float* pc_probs_flat = pc_probs.contiguous().data_ptr<float>();
  float* img_cls_loss_weights_flat =
      img_cls_loss_weights.contiguous().data_ptr<float>();
  float* im_labels_flat = im_labels.contiguous().data_ptr<float>();

  float* output_flat = output.contiguous().data_ptr<float>();

  int batch_size = pcl_probs.size(0);
  int channels = pcl_probs.size(1);
  int num_positive = pc_labels.size(1);

  float eps = 1e-6;

  for (int c = 0; c < channels; c++) {
    output_flat[c] = 0;
    if (im_labels_flat[c] != 0) {
      if (c == 0) {
        for (int i = 0; i < batch_size; i++) {
          if (labels_flat[i] == 0) {
            output_flat[c] -= cls_loss_weights_flat[i] *
                log(fmaxf(prob_data_flat[i * channels + c], eps));
          }
        }
      } else {
        for (int i = 0; i < num_positive; i++) {
          if (pc_labels_flat[i] == c) {
            output_flat[c] -= img_cls_loss_weights_flat[i] *
                log(fmaxf(pc_probs_flat[i], eps));
          }
        }
      }
    }
  }
  return 1;
}

int pcl_loss_backward_cpu(
    at::Tensor& pcl_probs,
    at::Tensor& labels,
    at::Tensor& cls_loss_weights,
    at::Tensor& gt_assignment,
    at::Tensor& pc_labels,
    at::Tensor& pc_probs,
    at::Tensor& pc_count,
    at::Tensor& img_cls_loss_weights,
    at::Tensor& im_labels,
    at::Tensor& top_grad,
    at::Tensor& bottom_grad) {
  // Grab the input tensor
  float* prob_data_flat = pcl_probs.contiguous().data_ptr<float>();
  float* labels_flat = labels.contiguous().data_ptr<float>();
  float* cls_loss_weights_flat =
      cls_loss_weights.contiguous().data_ptr<float>();
  float* gt_assignment_flat = gt_assignment.contiguous().data_ptr<float>();
  float* pc_labels_flat = pc_labels.contiguous().data_ptr<float>();
  float* pc_probs_flat = pc_probs.contiguous().data_ptr<float>();
  float* pc_count_flat = pc_count.contiguous().data_ptr<float>();
  float* img_cls_loss_weights_flat =
      img_cls_loss_weights.contiguous().data_ptr<float>();
  float* im_labels_flat = im_labels.contiguous().data_ptr<float>();

  float* bottom_grad_flat = bottom_grad.contiguous().data_ptr<float>();

  int batch_size = pcl_probs.size(0);
  int channels = pcl_probs.size(1);

  float eps = 1e-5;

  for (int i = 0; i < batch_size; i++) {
    for (int c = 0; c < channels; c++) {
      bottom_grad_flat[i * channels + c] = 0;
      if (im_labels_flat[c] != 0) {
        if (c == 0) {
          if (labels_flat[i] == 0) {
            bottom_grad_flat[i * channels + c] = -cls_loss_weights_flat[i] /
                fmaxf(prob_data_flat[i * channels + c], eps);
          }
        } else {
          if (labels_flat[i] == c) {
            int pc_index = gt_assignment_flat[i];
            if (c != pc_labels_flat[pc_index]) {
              printf("labels mismatch.\n");
            }
            bottom_grad_flat[i * channels + c] =
                -img_cls_loss_weights_flat[pc_index] /
                fmaxf(pc_count_flat[pc_index] * pc_probs_flat[pc_index], eps);
          }
        }
      }
    }
  }
  return 1;
}

} // namespace wsl
