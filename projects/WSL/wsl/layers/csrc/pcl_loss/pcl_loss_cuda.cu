#include <torch/types.h>

#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <math.h>

int PCLLossForwardLaucher(
    const float* bottom_data,
    const float* labels,
    const float* cls_loss_weights,
    const float* pc_labels,
    const float* pc_probs,
    const float* img_cls_loss_weights,
    const float* im_labels,
    const int batch_size,
    const int channels,
    const int num_positive,
    float* top_data,
    cudaStream_t stream);

int PCLLossBackwardLaucher(
    const float* top_diff,
    const float* prob_data,
    const float* labels,
    const float* cls_loss_weights,
    const float* gt_assignment,
    const float* pc_labels,
    const float* pc_probs,
    const float* pc_count,
    const float* img_cls_loss_weights,
    const float* im_labels,
    const int batch_size,
    const int channels,
    float* bottom_diff,
    cudaStream_t stream);

int pcl_loss_forward_cuda(
    at::Tensor& pcl_probs,
    at::Tensor& labels,
    at::Tensor& cls_loss_weights,
    at::Tensor& pc_labels,
    at::Tensor& pc_probs,
    at::Tensor& img_cls_loss_weights,
    at::Tensor& im_labels,
    at::Tensor& output) {
  at::cuda::CUDAGuard device_guard(pcl_probs.device());
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

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  PCLLossForwardLaucher(
      prob_data_flat,
      labels_flat,
      cls_loss_weights_flat,
      pc_labels_flat,
      pc_probs_flat,
      img_cls_loss_weights_flat,
      im_labels_flat,
      batch_size,
      channels,
      num_positive,
      output_flat,
      stream);

  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());
  return 1;
}

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
    at::Tensor& bottom_grad) {
  at::cuda::CUDAGuard device_guard(pcl_probs.device());
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

  float* top_grad_flat = top_grad.contiguous().data_ptr<float>();

  float* bottom_grad_flat = bottom_grad.contiguous().data_ptr<float>();

  int batch_size = pcl_probs.size(0);
  int channels = pcl_probs.size(1);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  PCLLossBackwardLaucher(
      top_grad_flat,
      prob_data_flat,
      labels_flat,
      cls_loss_weights_flat,
      gt_assignment_flat,
      pc_labels_flat,
      pc_probs_flat,
      pc_count_flat,
      img_cls_loss_weights_flat,
      im_labels_flat,
      batch_size,
      channels,
      bottom_grad_flat,
      stream);

  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());
  return 1;
}
