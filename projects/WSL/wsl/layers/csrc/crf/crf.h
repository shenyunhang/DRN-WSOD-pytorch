#pragma once
#include <torch/types.h>
#include "densecrf.h"

namespace wsl {

at::Tensor crf_forward_cpu(
    const at::Tensor& U,
    const at::Tensor& I,
    const int max_iter,
    const int scale_factor,
    const int color_factor,
    const float size_std,
    const float pos_w,
    const float pos_x_std,
    const float pos_y_std,
    const float bi_w,
    const float bi_x_std,
    const float bi_y_std,
    const float bi_r_std,
    const float bi_g_std,
    const float bi_b_std,
    const bool debug_info);

// Interface for Python
inline at::Tensor crf_forward(
    const at::Tensor& U,
    const at::Tensor& I,
    const int max_iter = 10,
    const int scale_factor = 1,
    const int color_factor = 13,
    const float size_std = 500,
    const float pos_w = 3,
    const float pos_x_std = 3,
    const float pos_y_std = 3,
    const float bi_w = 10,
    const float bi_x_std = 80,
    const float bi_y_std = 80,
    const float bi_r_std = 13,
    const float bi_g_std = 13,
    const float bi_b_std = 13,
    const bool debug_info = false) {
  if (I.device().type() == at::kCUDA) {
    AT_ERROR("Not compiled with GPU support");
  }

  return crf_forward_cpu(
      U,
      I,
      max_iter,
      scale_factor,
      color_factor,
      size_std,
      pos_w,
      pos_x_std,
      pos_y_std,
      bi_w,
      bi_x_std,
      bi_y_std,
      bi_r_std,
      bi_g_std,
      bi_b_std,
      debug_info);
}

} // namespace wsl
