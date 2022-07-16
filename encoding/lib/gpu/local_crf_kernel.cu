#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

#include "common.h"
#include "device_tensor.h"

template <typename DType>
__global__ void Local_CRF_Forward_kernel(
	DeviceTensor<DType, 4> output,
	DeviceTensor<DType, 4> input_F,
	DeviceTensor<DType, 4> input_Q,
	int kernel_len) {
	//  input_Q: B, C, H, W   input_F: B, kernel_size, H, W
	// output: B, C, H - kernel_len +1, W - kernel_len + 1
	int b = blockIdx.x;
	int C = input_Q.getSize(1);
	int H = input_Q.getSize(2);
	int W = input_Q.getSize(3);
	int kernel_size = kernel_len * kernel_len;
	int pad_len = (kernel_len - 1) / 2;
	int kernel_center_index = (kernel_size - 1) / 2;
	DType temp;
	// ignore the edge points
	// the edge points of input_F and input_Q are padding area (0).
	for (int x = threadIdx.x; x < (H - kernel_len + 1) * (W - kernel_len + 1); x += blockDim.x) {
		int h_output = x / (W - kernel_len + 1);
		int w_output = x % (W - kernel_len + 1);
		int h_input = h_output + pad_len;
		int w_input = w_output + pad_len;
		for (int c = 0; c < C; ++c) {
			temp = output[b][c][h_output][w_output];
			for (int index = 0; index < kernel_size; ++index) {
				if (index != kernel_center_index) {
					int i = index / kernel_len - pad_len;
					int j = index % kernel_len - pad_len;
					temp += input_F[b][index][h_input][w_input] * input_Q[b][c][h_input + i][w_input + j];
				}
			}
			output[b][c][h_output][w_output] = temp;
		}
	}
}


template <typename DType>
__global__ void Local_CRF_Backward_kernel(
	DeviceTensor<DType, 4> gradoutput,
	DeviceTensor<DType, 4> input_F,
	DeviceTensor<DType, 4> gradinput_F,
	DeviceTensor<DType, 4> input_Q,
	DeviceTensor<DType, 4> gradinput_Q,
	int kernel_len) {
	//  input_Q: B, C, H, W   input_F: B, kernel_size, H, W
	// gradoutput: B, C, H, W (padded)
	int b = blockIdx.x;
	int C = input_Q.getSize(1);
	int H = input_Q.getSize(2);
	int W = input_Q.getSize(3);
	int kernel_size = kernel_len * kernel_len;
	int pad_len = (kernel_len - 1) / 2;
	int kernel_center_index = (kernel_size - 1) / 2;
	DType temp;
	// ignore the edge points
	// the edge points of input_F and input_Q are padding area (0).
	// gradoutput must have been padded.    * important
	for (int x = threadIdx.x; x < (H - kernel_len + 1) * (W - kernel_len + 1); x += blockDim.x) {
		int h = x / (W - kernel_len + 1) + pad_len;
		int w = x % (W - kernel_len + 1) + pad_len;
		// grad for gradinput_F: B, kernel_size, H, W
		for (int index = 0; index < kernel_size; ++index) {
			if (index != kernel_center_index) {
				int i = index / kernel_len - pad_len;
				int j = index % kernel_len - pad_len;
				temp = gradinput_F[b][index][h][w];
				for (int c = 0; c < C; ++c) {
					temp += gradoutput[b][c][h][w] * input_Q[b][c][h + i][w + j];
				}
				gradinput_F[b][index][h][w] = temp;
			}
		}
		// grad for gradinput_Q: B, C, H, W
		for (int c = 0; c < C; ++c) {
			temp = gradinput_Q[b][c][h][w];
			for (int index = 0; index < kernel_size; ++index) {
				if (index != kernel_center_index) {
					int i = index / kernel_len - pad_len;
					int j = index % kernel_len - pad_len;
					temp += gradoutput[b][c][h - i][w - j] * input_F[b][index][h - i][w - j];
				}
			}
			gradinput_Q[b][c][h][w] = temp;
		}
	}
}


at::Tensor Local_CRF_Forward_CUDA(
	const at::Tensor input_F_,
	const at::Tensor input_Q_) {
	int B = input_Q_.size(0);
	int C = input_Q_.size(1);
	int H = input_Q_.size(2);
	int W = input_Q_.size(3);
	int kernel_size = input_F_.size(1);
	int kernel_len = sqrt(kernel_size);
	auto output_ = at::zeros({ B, C, H - kernel_len + 1, W - kernel_len + 1 }).to(input_Q_.device(), input_Q_.dtype());
	//auto output_ = at::zeros_like(input_Q_);
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	dim3 blocks(B);
	dim3 threads(getNumThreads((H - kernel_len + 1) * (W - kernel_len + 1)));
	AT_DISPATCH_FLOATING_TYPES(input_Q_.scalar_type(), "Local_CRF_Forward_CUDA", ([&] {
		/* Device tensors */
		DeviceTensor<scalar_t, 4> output = devicetensor<scalar_t, 4>(output_);
		DeviceTensor<scalar_t, 4> input_F = devicetensor<scalar_t, 4>(input_F_);
		DeviceTensor<scalar_t, 4> input_Q = devicetensor<scalar_t, 4>(input_Q_);
		/* kernel function */
		Local_CRF_Forward_kernel<scalar_t> << <blocks, threads, 0, stream >> > (
			output, input_F, input_Q, kernel_len);
	}));
	AT_ASSERT(cudaGetLastError() == cudaSuccess);
	return output_;
}

std::vector<at::Tensor> Local_CRF_Backward_CUDA(
	const at::Tensor gradoutput_,
	const at::Tensor input_F_,
	const at::Tensor input_Q_) {
	/* outputs*/
	int B = input_Q_.size(0);
	int C = input_Q_.size(1);
	int H = input_Q_.size(2);
	int W = input_Q_.size(3);
	int kernel_size = input_F_.size(1);
	int kernel_len = sqrt(kernel_size);
	auto gradinput_F_ = at::zeros_like(input_F_);
	auto gradinput_Q_ = at::zeros_like(input_Q_);
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	dim3 blocks(B);
	dim3 threads(getNumThreads((H - kernel_len + 1) * (W - kernel_len + 1)));
	AT_DISPATCH_FLOATING_TYPES(input_Q_.scalar_type(), "Local_CRF_Backward_CUDA", ([&] {
		/* Device tensors */
		DeviceTensor<scalar_t, 4> gradoutput = devicetensor<scalar_t, 4>(gradoutput_);
		DeviceTensor<scalar_t, 4> input_F = devicetensor<scalar_t, 4>(input_F_);
		DeviceTensor<scalar_t, 4> gradinput_F = devicetensor<scalar_t, 4>(gradinput_F_);
		DeviceTensor<scalar_t, 4> input_Q = devicetensor<scalar_t, 4>(input_Q_);
		DeviceTensor<scalar_t, 4> gradinput_Q = devicetensor<scalar_t, 4>(gradinput_Q_);
		/* kernel function */
		Local_CRF_Backward_kernel<scalar_t>
			<< <blocks, threads, 0, stream >> > (
				gradoutput, input_F, gradinput_F, input_Q, gradinput_Q, kernel_len);
	}));
	AT_ASSERT(cudaGetLastError() == cudaSuccess);
	return { gradinput_F_, gradinput_Q_ };
}