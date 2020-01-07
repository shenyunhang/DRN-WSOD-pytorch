#include <math.h>

#include <ATen/TensorUtils.h>
#include "crf.h"

namespace wsl {

template <typename T>
void bilinear_interpolation(
    const float* input,
    float* output,
    const int batch_size,
    const int num_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width) {
  int channels = num_channels * batch_size;

  const float rheight = (output_height > 1)
      ? (float)(input_height - 1) / (output_height - 1)
      : 0.f;
  const float rwidth =
      (output_width > 1) ? (float)(input_width - 1) / (output_width - 1) : 0.f;
  for (int h2 = 0; h2 < output_height; ++h2) {
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < input_height - 1) ? 1 : 0;
    const float h1lambda = h1r - h1;
    const float h0lambda = (float)1. - h1lambda;
    for (int w2 = 0; w2 < output_width; ++w2) {
      const float w1r = rwidth * w2;
      const int w1 = w1r;
      const int w1p = (w1 < input_width - 1) ? 1 : 0;
      const float w1lambda = w1r - w1;
      const float w0lambda = (float)1. - w1lambda;
      const float* Xdata = &input[h1 * input_width + w1];
      float* Ydata = &output[h2 * output_width + w2];
      for (int c = 0; c < channels; ++c) {
        Ydata[0] = h0lambda * (w0lambda * Xdata[0] + w1lambda * Xdata[w1p]) +
            h1lambda *
                (w0lambda * Xdata[h1p * input_width] +
                 w1lambda * Xdata[h1p * input_width + w1p]);
        Xdata += input_width * input_height;
        Ydata += output_width * output_height;
      }
    }
  }
}

template <typename T>
void image_process(
    const float* input,
    unsigned char* output,
    const int batch_size,
    const int height,
    const int width) {
  // TODO(YH): add argument
  float mean[] = {102.9801, 115.9465, 122.7717};
  for (int b = 0; b < batch_size; b++) {
    for (int c = 0; c < 3; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          int idx_i = ((b * 3 + c) * height + h) * width + w;
          int idx_o = ((b * height + h) * width + w) * 3 + c;
          output[idx_o] = (unsigned char)(input[idx_i] + mean[c]);
        }
      }
    }
  }
}

template <typename T>
void unary_process(
    const float* input,
    float* output,
    const int batch_size,
    const int num_classes,
    const int height,
    const int width) {
  const float min_prob = 0.0001;
  for (int b = 0; b < batch_size; b++) {
    for (int c = 0; c < num_classes; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          int idx_i = ((b * num_classes + c) * height + h) * width + w;
          int idx_o = ((b * height + h) * width + w) * num_classes + c;
          output[idx_o] = std::max(input[idx_i], min_prob);
          // output[idx_o] = -1. * std::max(input[idx_i], min_prob);
          // output[idx_o] = -1. * input[idx_i];
        }
      }
    }
  }
}

template <typename T>
void result_process(
    const float* input,
    float* output,
    const int batch_size,
    const int num_classes,
    const int height,
    const int width,
    const at::Tensor& I) {
  const float min_prob = 0.0001;

  at::Tensor N = at::zeros({batch_size, height, width}, I.options());
  float* Nmdata = N.contiguous().data_ptr<float>();

  for (int b = 0; b < batch_size; b++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        Nmdata[0] = 0;
        for (int c = 0; c < num_classes; c++) {
          int idx_i = ((b * height + h) * width + w) * num_classes + c;
          int idx_o = ((b * num_classes + c) * height + h) * width + w;
          output[idx_o] = std::max(input[idx_i], min_prob);
          Nmdata[0] += output[idx_o];
        }
        Nmdata += 1;
      }
    }
  }

  const float* Ndata = N.contiguous().data_ptr<float>();

  for (int b = 0; b < batch_size; b++) {
    for (int c = 0; c < num_classes; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          int idx_o = ((b * num_classes + c) * height + h) * width + w;
          float norm = *(Ndata + (b * height + h) * width + w);
          // output[idx_o] = log(output[idx_o] / norm);
          output[idx_o] = output[idx_o] / norm;
        }
      }
    }
  }
}

template <typename T>
class DenseCRFOp {
 public:
  DenseCRFOp(
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
    max_iter_ = max_iter;
    scale_factor_ = scale_factor;
    color_factor_ = color_factor;
    SIZE_STD = size_std;
    POS_W = pos_w;
    POS_X_STD = pos_x_std;
    POS_Y_STD = pos_y_std;
    BI_W = bi_w;
    BI_X_STD = bi_x_std;
    BI_Y_STD = bi_y_std;
    BI_R_STD = bi_r_std;
    BI_G_STD = bi_g_std;
    BI_B_STD = bi_b_std;
    debug_info_ = debug_info;
  }
  ~DenseCRFOp() {}

  at::Tensor RunOnDevice(const at::Tensor& U, const at::Tensor& I);

  void set_unary_energy(const float* unary_costs_ptr);

  void add_pairwise_energy(
      float w1,
      float theta_alpha_1,
      float theta_alpha_2,
      float theta_betta_1,
      float theta_betta_2,
      float theta_betta_3,
      float w2,
      float theta_gamma_1,
      float theta_gamma_2,
      const unsigned char* im);

  void map(int n_iters, int* result);
  void inference(int n_iters, float* result);

  int npixels();
  int nlabels();

  void
  dense_crf(const unsigned char* image, const float* unary, float* probs_out);

 protected:
  int max_iter_;
  int scale_factor_;
  int color_factor_;

  float SIZE_STD;

  float POS_W;
  float POS_X_STD;
  float POS_Y_STD;
  float BI_W;
  float BI_X_STD;
  float BI_Y_STD;
  float BI_R_STD;
  float BI_G_STD;
  float BI_B_STD;

  bool debug_info_;

  DenseCRF2D* m_crf;

  int H;
  int W;
  int m_nlabels;
};

template <>
int DenseCRFOp<float>::npixels() {
  return W * H;
}

template <>
int DenseCRFOp<float>::nlabels() {
  return m_nlabels;
}

template <>
void DenseCRFOp<float>::add_pairwise_energy(
    float w1,
    float theta_alpha_1,
    float theta_alpha_2,
    float theta_betta_1,
    float theta_betta_2,
    float theta_betta_3,
    float w2,
    float theta_gamma_1,
    float theta_gamma_2,
    const unsigned char* im) {
  m_crf->addPairwiseGaussian(
      theta_gamma_1, theta_gamma_2, new PottsCompatibility(w2));
  m_crf->addPairwiseBilateral(
      theta_alpha_1,
      theta_alpha_2,
      theta_betta_1,
      theta_betta_2,
      theta_betta_3,
      im,
      new PottsCompatibility(w1));
  // m_crf->addPairwiseGaussian(3, 3, new PottsCompatibility(3));
  // m_crf->addPairwiseBilateral(80, 80, 13, 13, 13, im,
  // new PottsCompatibility(10));
}

template <>
void DenseCRFOp<float>::set_unary_energy(const float* unary_costs_ptr) {
  m_crf->setUnaryEnergy(
      Eigen::Map<const Eigen::MatrixXf>(unary_costs_ptr, m_nlabels, W * H));
}

template <>
void DenseCRFOp<float>::map(int n_iters, int* labels) {
  VectorXs labels_vec = m_crf->map(n_iters);
  for (int i = 0; i < (W * H); ++i)
    labels[i] = labels_vec(i);
}

template <>
void DenseCRFOp<float>::inference(int n_iters, float* probs_out) {
  MatrixXf probs = m_crf->inference(n_iters);
  for (int i = 0; i < npixels(); ++i)
    for (int j = 0; j < nlabels(); ++j)
      probs_out[i * nlabels() + j] = probs(j, i);
}

template <>
void DenseCRFOp<float>::dense_crf(
    const unsigned char* image,
    const float* unary,
    float* probs_out) {
  // set unary potentials
  set_unary_energy(unary);

  // set pairwise potentials
  // add_pairwise_energy(10, 80 / scale_factor_, 80 / scale_factor_,
  // color_factor_, color_factor_, color_factor_, 3, 3 / scale_factor_, 3 /
  // scale_factor_, image);
  add_pairwise_energy(
      BI_W,
      BI_X_STD / scale_factor_,
      BI_Y_STD / scale_factor_,
      BI_R_STD,
      BI_G_STD,
      BI_B_STD,
      POS_W,
      POS_X_STD / scale_factor_,
      POS_Y_STD / scale_factor_,
      image);

  // run inference
  inference(max_iter_, probs_out);
}

template <>
at::Tensor DenseCRFOp<float>::RunOnDevice(
    const at::Tensor& U,
    const at::Tensor& I) {
  AT_ASSERTM(!U.type().is_cuda(), "U must be a CPU tensor");
  AT_ASSERTM(!I.type().is_cuda(), "I must be a CPU tensor");

  // Grab the input tensor
  float* U_flat = U.contiguous().data_ptr<float>();
  float* I_flat = I.contiguous().data_ptr<float>();

  TORCH_CHECK(
      U.ndimension() == 4,
      "4D weight tensor (batch_size, num_class, height, width) expected, "
      "but got: %d",
      U.ndimension());
  TORCH_CHECK(
      I.ndimension() == 4,
      "4D weight tensor (batch_size, num_class, height, width) expected, "
      "but got: %d",
      I.ndimension());
  TORCH_CHECK(
      U.size(0) == I.size(0),
      "The 1st dimension of U and I should be the same, but got U.size(0): %d I.size(0): %d",
      U.size(0),
      I.size(0));
  TORCH_CHECK(
      I.size(1) == 3,
      "The 3rd dimension of size 3 expected, "
      "but got: %d",
      I.size(1));

  int batch_size = U.size(0);
  int num_classes = U.size(1);
  int height = U.size(2);
  int width = U.size(3);
  int height_im = I.size(2);
  int width_im = I.size(3);
  int H = height;
  int W = width;
  int m_nlabels = num_classes;

  m_crf = new DenseCRF2D(W, H, m_nlabels);

  at::Tensor M =
      at::zeros({batch_size, num_classes, height, width}, I.options());
  float* M_flat = M.contiguous().data_ptr<float>();

  at::Tensor MT =
      at::zeros({batch_size, height, width, num_classes}, M.options());

   at::Tensor IT = at::zeros({batch_size, height, width, 3},
   I.options().dtype(at::kByte));

  if (height != height_im || width != width_im) {
    at::Tensor IB = at::zeros({batch_size, 3, height, width}, I.options());
    bilinear_interpolation<float>(
        I_flat,
        IB.contiguous().data_ptr<float>(),
        batch_size,
        3,
        height_im,
        width_im,
        height,
        width);

    image_process<float>(
        IB.contiguous().data_ptr<float>(),
        IT.contiguous().data_ptr<unsigned char>(),
        batch_size,
        height,
        width);
  } else {
    image_process<float>(
        I_flat,
        IT.contiguous().data_ptr<unsigned char>(),
        batch_size,
        height,
        width);
  }

  at::Tensor UT =
      at::zeros({batch_size, height, width, num_classes}, I.options());
  unary_process<float>(
      U_flat,
      UT.contiguous().data_ptr<float>(),
      batch_size,
      num_classes,
      height,
      width);

  for (int b = 0; b < batch_size; b++) {
    const unsigned char* image =
        IT.contiguous().data_ptr<unsigned char>() + b * height * width * 3;
    const float* unary =
        UT.contiguous().data_ptr<float>() + b * height * width * num_classes;
    float* probs_out =
        MT.contiguous().data_ptr<float>() + b * height * width * num_classes;

    // auto adjust scale_factor_
    scale_factor_ = 1.0 * SIZE_STD / std::max(height, width);

    dense_crf(image, unary, probs_out);
  }

  result_process<float>(
      MT.contiguous().data_ptr<float>(),
      M_flat,
      batch_size,
      num_classes,
      height,
      width,
      I);

  delete m_crf;
  return M;
}

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
    const bool debug_info) {
  DenseCRFOp<float> op = DenseCRFOp<float>(
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
      debug_info

  );
  at::Tensor M = op.RunOnDevice(U, I);

  return M;
}
} // namespace wsl
