// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#include <torch/extension.h>
#include "crf/crf.h"
#include "csc/csc.h"
#include "pcl_loss/pcl_loss.h"

namespace wsl {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pcl_loss_forward", &pcl_loss_forward, "pcl_loss_forward");
  m.def("pcl_loss_backward", &pcl_loss_backward, "pcl_loss_backward");

  m.def("csc_forward", &csc_forward, "csc_forward");

  m.def("crf_forward", &crf_forward, "crf_forward");
}

} // namespace wsl
