#pragma once
#include <torch/types.h>

namespace wsl {

int pcl_loss_forward_cpu(
    at::Tensor& pcl_probs,
    at::Tensor& labels,
    at::Tensor& cls_loss_weights,
    at::Tensor& pc_labels,
    at::Tensor& pc_probs,
    at::Tensor& img_cls_loss_weights,
    at::Tensor& im_labels,
    at::Tensor& output);

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
    at::Tensor& bottom_grad);

#ifdef WITH_CUDA
int pcl_loss_forward_cuda(
    at::Tensor& pcl_probs,
    at::Tensor& labels,
    at::Tensor& cls_loss_weights,
    at::Tensor& pc_labels,
    at::Tensor& pc_probs,
    at::Tensor& img_cls_loss_weights,
    at::Tensor& im_labels,
    at::Tensor& output);

int pcl_loss_backward_cuda(
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
    at::Tensor& bottom_grad);
#endif

// Interface for Python
inline int pcl_loss_forward(
    at::Tensor& pcl_probs,
    at::Tensor& labels,
    at::Tensor& cls_loss_weights,
    at::Tensor& pc_labels,
    at::Tensor& pc_probs,
    at::Tensor& img_cls_loss_weights,
    at::Tensor& im_labels,
    at::Tensor& output) {
  if (output.type().is_cuda() && false) {
#ifdef WITH_CUDA
    return pcl_loss_forward_cuda(
        pcl_probs,
        labels,
        cls_loss_weights,
        pc_labels,
        pc_probs,
        img_cls_loss_weights,
        im_labels,
        output);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return pcl_loss_forward_cpu(
      pcl_probs,
      labels,
      cls_loss_weights,
      pc_labels,
      pc_probs,
      img_cls_loss_weights,
      im_labels,
      output);
}

inline int pcl_loss_backward(
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
  if (bottom_grad.type().is_cuda() && false) {
#ifdef WITH_CUDA
    return pcl_loss_backward_cuda(
        pcl_probs,
        labels,
        cls_loss_weights,
        gt_assignment,
        pc_labels,
        pc_probs,
        pc_count,
        img_cls_loss_weights,
        im_labels,
        top_grad,
        bottom_grad);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return pcl_loss_backward_cpu(
      pcl_probs,
      labels,
      cls_loss_weights,
      gt_assignment,
      pc_labels,
      pc_probs,
      pc_count,
      img_cls_loss_weights,
      im_labels,
      top_grad,
      bottom_grad);
}

} // namespace wsl
