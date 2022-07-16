#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

at::Tensor BatchNorm_Forward_CUDA(
  const at::Tensor input_, 
  const at::Tensor mean_,
  const at::Tensor std_,
  const at::Tensor gamma_,
  const at::Tensor beta_,
  float eps);

std::vector<at::Tensor> BatchNorm_Backward_CUDA(
  const at::Tensor gradoutput_,
  const at::Tensor input_,
  const at::Tensor ex_, 
  const at::Tensor exs_,
  const at::Tensor gamma_,
  const at::Tensor beta_,
  float eps);

std::vector<at::Tensor> Expectation_Forward_CUDA(
  const at::Tensor input_);

at::Tensor Expectation_Backward_CUDA(
  const at::Tensor input_,
  const at::Tensor gradEx_,
  const at::Tensor gradExs_);

at::Tensor Local_CRF_Forward_CUDA(
	const at::Tensor input_F_,
	const at::Tensor input_Q_);

std::vector<at::Tensor> Local_CRF_Backward_CUDA(
	const at::Tensor gradoutput_,
	const at::Tensor input_F_,
	const at::Tensor input_Q_);

at::Tensor Local_Diff_Forward_CUDA(
	const at::Tensor input_F_,
	int kernel_len);

at::Tensor Local_Diff_Backward_CUDA(
	const at::Tensor gradoutput_,
	const at::Tensor input_F_,
	int kernel_len);