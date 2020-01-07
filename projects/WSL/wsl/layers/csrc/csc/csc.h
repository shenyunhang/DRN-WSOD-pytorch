#pragma once
#include <torch/types.h>

namespace wsl {

#ifdef WITH_CUDA
at::Tensor csc_forward_cuda(
    const at::Tensor& cpgs,
    const at::Tensor& labels,
    const at::Tensor& preds,
    const at::Tensor& rois,
    const float tau_,
    const bool debug_info_,
    const float fg_threshold_,
    const float mass_threshold_,
    const float density_threshold_,
    const bool area_sqrt_,
    const float context_scale_);
#endif

// Interface for Python
inline at::Tensor csc_forward(
    const at::Tensor& cpgs,
    const at::Tensor& labels,
    const at::Tensor& preds,
    const at::Tensor& rois,
    const float tau_ = 0.7,
    const bool debug_info_ = false,
    const float fg_threshold_ = 0.1,
    const float mass_threshold_ = 0.2,
    const float density_threshold_ = 0.0,
    const bool area_sqrt_ = true,
    const float context_scale_ = 1.8) {
  if (cpgs.device().type() == at::kCUDA) {
#ifdef WITH_CUDA
    return csc_forward_cuda(
        cpgs,
        labels,
        preds,
        rois,
        tau_,
        debug_info_,
        fg_threshold_,
        mass_threshold_,
        density_threshold_,
        area_sqrt_,
        context_scale_);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not compiled with CPU support");
}

} // namespace wsl
