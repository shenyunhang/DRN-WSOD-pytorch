#include <cfloat>
#include <functional>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

const float kMIN_SCORE = -1.0 * 1e20;

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

namespace {

const int CUDA_NUM_THREADS = 1024;
const int kMaxGridNum = 65535;

inline int GET_BLOCKS(const int N) {
  return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

} // namespace

template <typename T>
__global__ void kernel_show(
    const T* Xdata,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int ndim,
    const int gpu_id,
    const int uuid) {
  printf(
      "uuid=%d gpu=%d ndim=%d b = %d c = %d h = %d w = %d\n",
      uuid,
      gpu_id,
      ndim,
      batch_size,
      channels,
      height,
      width);
  for (int b = 0; b < batch_size; b++) {
    for (int c = 0; c < channels; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          int index_X = ((b * channels + c) * height + h) * width + w;
          printf(
              "b = %d c = %d h = %d w = %d %.32f\n",
              b,
              c,
              h,
              w,
              Xdata[index_X]);
        }
      }
    }
  }
}

template <typename T>
__global__ void kernel_show_c(
    const T* Xdata,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int ndim,
    const int gpu_id,
    const int uuid,
    const int c) {
  printf(
      "uuid=%d gpu=%d ndim=%d b = %d c = %d h = %d w = %d\n",
      uuid,
      gpu_id,
      ndim,
      batch_size,
      channels,
      height,
      width);
  for (int b = 0; b < batch_size; b++) {
    // for (int c = 0; c < channels; c++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        int index_X = ((b * channels + c) * height + h) * width + w;
        printf(
            "b = %d c = %d h = %d w = %d %.32f\n", b, c, h, w, Xdata[index_X]);
      }
    }
    //}
  }
}

template <typename T>
__global__ void binary_kernel(
    const int nthreads,
    const T* const x,
    T* const y,
    const T threshold) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    if (x[index] >= threshold) {
      y[index] = 1;
    } else {
      y[index] = 0;
    }
  }
}

template <typename T>
void integral_cpu(const T* src, T* sum, const int height, const int width) {
  T s = 0;
  for (int x = 0; x < width; x++) {
    s += src[x];
    sum[x] = s;
  }
  src += width;
  sum += width;
  for (int y = 1; y < height; y++, src += width, sum += width) {
    s = 0;
    for (int x = 0; x < width; x++) {
      s += src[x];
      sum[x] = sum[x - width] + s;
    }
  }
}

template <typename T>
void binary_and_integral_cpu(
    const T* src,
    T* sum,
    const int height,
    const int width,
    const T threshold) {
  T s = 0;
  for (int x = 0; x < width; x++) {
    if (src[x] >= threshold) {
      s += 1;
    } else {
      s += 0;
    }
    sum[x] = s;
  }
  src += width;
  sum += width;
  for (int y = 1; y < height; y++, src += width, sum += width) {
    s = 0;
    for (int x = 0; x < width; x++) {
      if (src[x] >= threshold) {
        s += 1;
      } else {
        s += 0;
      }
      sum[x] = sum[x - width] + s;
    }
  }
}

template <typename T>
T get_sum(const int N, const T* data) {
  T sum_val = 0;
  for (int i = 0; i < N; i++) {
    sum_val += *data;
    data += 1;
  }
  return sum_val;
}

template <typename T>
T get_max(const int N, const T* data) {
  T max_val = -FLT_MAX;
  for (int i = 0; i < N; i++) {
    if (*data > max_val) {
      max_val = *data;
    }
    data += 1;
  }
  return max_val;
}

template <typename T>
__global__ void CSCPool(
    const int nthreads,
    const T* cpg_data,
    const int height_im,
    const int width_im,
    const T* rois_data,
    const int num_class,
    const int cls_id,
    const T min_density,
    const T min_mass,
    const bool area_sqrt,
    const T context_scale,
    T* const top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int rois_index = index;

    rois_data += 5 * rois_index;
    int wstart = round(rois_data[1]);
    int hstart = round(rois_data[2]);
    int wend = round(rois_data[3]);
    int hend = round(rois_data[4]);

    // if (wstart < 0) wstart = 0;
    // if (wstart >= width_im) wstart = width_im - 1;
    // if (hstart < 0) hstart = 0;
    // if (hstart >= height_im) hstart = height_im - 1;

    // Check RoI
    // if (wstart >= 0 && hstart >= 0 && wstart < wend && hstart < hend &&
    // wend < width_im && hend < height_im) {
    //} else {
    // top_data[rois_index * num_class + cls_id] = kMIN_SCORE;
    //// 这里面是for循环，用return会中断后续的循环
    // continue;
    //}
    wstart = max(min(wstart, width_im - 1), 0);
    hstart = max(min(hstart, height_im - 1), 0);
    wend = max(min(wend, width_im - 1), 0);
    hend = max(min(hend, height_im - 1), 0);

    // caculate the inner and outer RoI coordinate
    T width_roi = wend - wstart;
    T height_roi = hend - hstart;
    // T context_scale = 1.8;
    // T context_scale = sqrtf(2.0);
    T width_roi_inner = 1.0 * width_roi / context_scale;
    T height_roi_inner = 1.0 * height_roi / context_scale;
    T width_roi_outer = 1.0 * width_roi * context_scale;
    T height_roi_outer = 1.0 * height_roi * context_scale;
    T wcenter = 1.0 * (wend + wstart) / 2.0;
    T hcenter = 1.0 * (hend + hstart) / 2.0;

    int wstart_inner = round(wcenter - width_roi_inner / 2.0);
    int hstart_inner = round(hcenter - height_roi_inner / 2.0);
    int wend_inner = round(wcenter + width_roi_inner / 2.0);
    int hend_inner = round(hcenter + height_roi_inner / 2.0);

    int wstart_outer = round(max(wcenter - width_roi_outer / 2.0, 0.0));
    int hstart_outer = round(max(hcenter - height_roi_outer / 2.0, 0.0));
    int wend_outer =
        round(min(wcenter + width_roi_outer / 2.0, width_im - 1.0));
    int hend_outer =
        round(min(hcenter + height_roi_outer / 2.0, height_im - 1.0));

    width_roi = wend - wstart + 1;
    height_roi = hend - hstart + 1;
    width_roi_inner = wend_inner - wstart_inner + 1;
    height_roi_inner = hend_inner - hstart_inner + 1;
    width_roi_outer = wend_outer - wstart_outer + 1;
    height_roi_outer = hend_outer - hstart_outer + 1;

    // a1-a2-a3+a4
    T a1, a2, a3, a4;

    // CPG sum of RoI
    a1 = cpg_data[hend * width_im + wend];
    a2 = (wstart - 1 >= 0) ? cpg_data[hend * width_im + (wstart - 1)] : 0;
    a3 = (hstart - 1 >= 0) ? cpg_data[(hstart - 1) * width_im + wend] : 0;
    a4 = (hstart - 1 >= 0 && wstart - 1 >= 0)
        ? cpg_data[(hstart - 1) * width_im + (wstart - 1)]
        : 0;
    T sum_roi = a1 - a2 - a3 + a4;

    // CPG sum of inner RoI
    a1 = cpg_data[hend_inner * width_im + wend_inner];
    a2 = (wstart_inner - 1 >= 0)
        ? cpg_data[hend_inner * width_im + (wstart_inner - 1)]
        : 0;
    a3 = (hstart_inner - 1 >= 0)
        ? cpg_data[(hstart_inner - 1) * width_im + wend_inner]
        : 0;
    a4 = (hstart_inner - 1 >= 0 && wstart_inner - 1 >= 0)
        ? cpg_data[(hstart_inner - 1) * width_im + (wstart_inner - 1)]
        : 0;
    T sum_inner = a1 - a2 - a3 + a4;

    // CPG sum of outer RoI
    a1 = cpg_data[hend_outer * width_im + wend_outer];
    a2 = (wstart_outer - 1 >= 0)
        ? cpg_data[hend_outer * width_im + (wstart_outer - 1)]
        : 0;
    a3 = (hstart_outer - 1 >= 0)
        ? cpg_data[(hstart_outer - 1) * width_im + wend_outer]
        : 0;
    a4 = (hstart_outer - 1 >= 0 && wstart_outer - 1 >= 0)
        ? cpg_data[(hstart_outer - 1) * width_im + (wstart_outer - 1)]
        : 0;
    T sum_outer = a1 - a2 - a3 + a4;

    // area size
    T area_roi = height_roi * width_roi;
    T area_inner = height_roi_inner * width_roi_inner;
    T area_outer = height_roi_outer * width_roi_outer;

    T area_frame = max(area_roi - area_inner, T(1));
    T area_context = max(area_outer - area_roi, T(1));

    //-----------------------------------------------------------------------
    T score;
    T sum_frame = sum_roi - sum_inner;
    T sum_context = sum_outer - sum_roi;

    // current best
    if (area_sqrt) {
      score = sum_frame / sqrt(area_frame) - sum_context / sqrt(area_context);
    } else {
      score = sum_frame / area_frame - sum_context / area_context;
    }

    // bad at test debug
    // T score = (sum_roi - sum_inner) - (sum_outer - sum_roi);

    // (msra 0223):
    // T score = ((sum_roi - 2.0 * (sum_outer - sum_roi)) *
    //(2.0 * (sum_roi - sum_inner) - sum_inner)) /
    // area_roi;
    // if ((sum_roi - 2.0 * (sum_outer - sum_roi)) < 0 &&
    //(2.0 * (sum_roi - sum_inner) - sum_inner) < 0) {
    // score = -1.0 * score;
    //}

    // (msra 0101): bad
    // T score = sqrt((sum_roi - sum_inner) / area_frame) -
    //               sqrt((sum_outer - sum_roi) / area_context);

    // (msra 12.30): very bad
    // T score =
    //    (sum_roi - sum_inner) / area_frame - (sum_outer - sum_roi) /
    // area_context;

    // (msra 12.29): bad
    // T score = ((sum_roi - sum_inner) - (sum_outer - sum_roi)) /
    // area_frame;

    // (msra 0105): bad than (msra 12.29)
    // T score = ((sum_roi - sum_inner) - (sum_outer - sum_roi)) /
    // sqrt(area_frame);

    //-----------------------------------------------------------------------

    // if (sum_roi < min_mass) score = kMIN_SCORE;

    top_data[rois_index * num_class + cls_id] = score;
  }
}

namespace wsl {

at::Tensor csc_forward_cuda(
    const at::Tensor& M,
    const at::Tensor& X,
    const at::Tensor& Y,
    const at::Tensor& R,
    const float tau_,
    const bool debug_info_,
    const float fg_threshold_,
    const float mass_threshold_,
    const float density_threshold_,
    const bool area_sqrt_,
    const float context_scale_) {
  CAFFE_ENFORCE_EQ(M.dim(), 4);
  CAFFE_ENFORCE_EQ(X.dim(), 2);
  CAFFE_ENFORCE_EQ(Y.dim(), 2);
  CAFFE_ENFORCE_EQ(R.dim(), 2);
  CAFFE_ENFORCE_EQ(X.size(0), Y.size(0));
  CAFFE_ENFORCE_EQ(X.size(0), M.size(0));
  CAFFE_ENFORCE_EQ(X.size(1), Y.size(1));
  CAFFE_ENFORCE_EQ(X.size(1), M.size(1));
  CAFFE_ENFORCE_EQ(R.size(1), 5);

  at::cuda::CUDAGuard device_guard(M.device());

  const int batch_size = X.size(0);
  const int num_classes = X.size(1);
  const int num_rois = R.size(0);
  const int cpg_height = M.size(2);
  const int cpg_width = M.size(3);

  at::Tensor W = at::ones({num_rois, num_classes}, M.options());

  const int gpu_id = M.device().index();
  int uuid;
  if (debug_info_) {
    srand(time(NULL));
    uuid = rand();
  }

  at::Tensor Xcpu = X.to(at::kCPU, /*non_blocking=*/false);
  const float* Xcpudata = Xcpu.contiguous().data_ptr<float>();

  at::Tensor Ycpu = Y.to(at::kCPU, /*non_blocking=*/false);
  const float* Ycpudata = Ycpu.contiguous().data_ptr<float>();

  for (int b = 0; b < batch_size; b++) {
    for (int c = 0; c < num_classes; c++) {
      int label_idx = b * num_classes + c;
      float label_value = Xcpudata[label_idx];
      float pred_value = Ycpudata[label_idx];
      if (debug_info_) {
        printf(
            "uuid %d gpu %d b %d c %d: %.32f %.32f\n",
            uuid,
            gpu_id,
            b,
            c,
            label_value,
            pred_value);
      }
      if (label_value < 0.5) {
        continue;
      }
      // if (pred_value < tau_) {
      // continue;
      //}

      // Get CPG map
      at::Tensor m = at::from_blob(
          M.contiguous().data_ptr<float>() + cpg_height * cpg_width * label_idx,
          {cpg_height, cpg_width},
          M.options());

      // Get max value
      at::Tensor mcpu = m.to(at::kCPU, /*non_blocking=*/false);
      // float max_val = get_max<float>(mcpu.numel(), mcpu.data<float>());
      float max_val = 1.;
      if (debug_info_) {
        printf("uuid %d gpu %d max_val %.32f\n", uuid, gpu_id, max_val);
      }

      float im_mass = 0;
      float im_density = 0;
      // im_mass = get_sum<float>(mcpu.numel(), mcpu.data<float>());
      // im_density = 1.0 * im_mass / cpg_height / cpg_width;
      if (debug_info_) {
        printf(
            "uuid %d gpu %d im_mass %.32f im_density %.32f\n",
            uuid,
            gpu_id,
            im_mass,
            im_density);
      }

      // Get Integral map
      at::Tensor icpu = at::zeros(mcpu.sizes(), mcpu.options());
      binary_and_integral_cpu<float>(
          mcpu.contiguous().data_ptr<float>(),
          icpu.contiguous().data_ptr<float>(),
          cpg_height,
          cpg_width,
          max_val * fg_threshold_);
      // CAFFE_ENFORCE_EQ(icpu.data<float>()[cpg_height * cpg_width - 1],
      // im_mass);
      if (debug_info_) {
        printf(
            "uuid %d gpu %d im_mass in icpu %.32f im_mass %.32f\n",
            uuid,
            gpu_id,
            icpu.data<float>()[cpg_height * cpg_width - 1],
            im_mass);
      }

      m = icpu.to(m.device(), /*non_blocking=*/false);

      // CSC Pooling
      cudaStream_t stream = at::cuda::getCurrentCUDAStream();
      dim3 grid(std::min(at::cuda::ATenCeilDiv(long(num_rois), 512L), 4096L));
      dim3 block(512);
      CSCPool<float><<<grid, block, 0, stream>>>(
          num_rois,
          m.contiguous().data_ptr<float>(),
          cpg_height,
          cpg_width,
          R.contiguous().data_ptr<float>(),
          num_classes,
          c,
          im_density * density_threshold_,
          im_mass * mass_threshold_,
          area_sqrt_,
          context_scale_,
          W.contiguous().data_ptr<float>());
      cudaDeviceSynchronize();
      AT_CUDA_CHECK(cudaGetLastError());

      at::Tensor Wcpu = W.to(at::kCPU, /*non_blocking=*/false);
      // normalization max value to |1|
      float* Wcpudata = Wcpu.contiguous().data_ptr<float>();
      float max_value = 0;
      float min_value = 0;
      for (int r = 0; r < num_rois; r++) {
        float value = Wcpudata[r * num_classes + c];
        if (value > max_value) {
          max_value = value;
        }
        if (value < min_value && value != kMIN_SCORE) {
          min_value = value;
        }
      }
      if (max_value > 0 && min_value < 0) {
        for (int r = 0; r < num_rois; r++) {
          float value = Wcpudata[r * num_classes + c];
          if (value == kMIN_SCORE) {
            value = -1;
          } else {
            value = value > 0 ? value / max_value : value / (-min_value);
          }
          // value = value > 0 ? value / max_value : -1;
          Wcpudata[r * num_classes + c] = value;
        }
      } else if (max_value > 0 && min_value == 0) {
        for (int r = 0; r < num_rois; r++) {
          float value = Wcpudata[r * num_classes + c];
          if (value == kMIN_SCORE) {
            value = -1;
          } else {
            value = value / max_value;
          }
          Wcpudata[r * num_classes + c] = value;
        }
      } else {
        for (int r = 0; r < num_rois; r++) {
          Wcpudata[r * num_classes + c] = 1.0;
        }
      }
      for (int r = 0; r < num_rois; r++) {
        Wcpudata[r * num_classes + c] =
            pred_value * Wcpudata[r * num_classes + c] + (1 - pred_value) * 1;
      }
      W = Wcpu.to(W.device(), /*non_blocking=*/false);

      if (debug_info_) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        kernel_show_c<float><<<GET_BLOCKS(1), 1, 0, stream>>>(
            W.contiguous().data_ptr<float>(),
            num_rois,
            num_classes,
            1,
            1,
            W.dim(),
            gpu_id,
            uuid,
            c);
        AT_CUDA_CHECK(cudaGetLastError());
      }
    }
  }
  AT_CUDA_CHECK(cudaGetLastError());

  return W;
}

} // namespace wsl
