#include "operator.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("batchnorm_forward", &BatchNorm_Forward_CUDA, "BatchNorm forward (CUDA)");
  m.def("batchnorm_backward", &BatchNorm_Backward_CUDA, "BatchNorm backward (CUDA)");
  m.def("expectation_forward", &Expectation_Forward_CUDA, "Expectation forward (CUDA)");
  m.def("expectation_backward", &Expectation_Backward_CUDA, "Expectation backward (CUDA)");
  m.def("local_crf_forward", &Local_CRF_Forward_CUDA, "Local CRF forward (CUDA)");
  m.def("local_crf_backward", &Local_CRF_Backward_CUDA, "Local CRF backward (CUDA)");
  m.def("local_diff_forward", &Local_Diff_Forward_CUDA, "Local difference forward (CUDA)");
  m.def("local_diff_backward", &Local_Diff_Backward_CUDA, "Local difference backward (CUDA)");
}
