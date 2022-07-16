#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

#include "common.h"
#include "device_tensor.h"

template <typename DType>
__global__ void Local_Diff_Forward_kernel(
	DeviceTensor<DType, 4> output,
	DeviceTensor<DType, 4> input_F,
	int kernel_len) {
	//  input_F: B, C, H, W
	// output: B, kernel_size, H - kernel_len +1, W - kernel_len + 1
	int b = blockIdx.x;
	int C = input_F.getSize(1);
	int H = input_F.getSize(2);
	int W = input_F.getSize(3);
	int kernel_size = kernel_len * kernel_len;
	int pad_len = (kernel_len - 1) / 2;
	int kernel_center_index = (kernel_size - 1) / 2;
	DType temp;
	DType diff;
	// ignore the edge points
	// the edge points of input_F are padding area (0).
	for (int x = threadIdx.x; x < (H - kernel_len + 1) * (W - kernel_len + 1); x += blockDim.x) {
		int h_output = x / (W - kernel_len + 1);
		int w_output = x % (W - kernel_len + 1);
		int h_input = h_output + pad_len;
		int w_input = w_output + pad_len;
		for (int index = 0; index < kernel_size; ++index) {
			if (index != kernel_center_index) {
				int i = index / kernel_len - pad_len;
				int j = index % kernel_len - pad_len;
				temp = output[b][index][h_output][w_output];
				for (int c = 0; c < C; ++c) {
					diff = input_F[b][c][h_input][w_input] - input_F[b][c][h_input + i][w_input + j];
					temp += diff * diff;
				}
				output[b][index][h_output][w_output] = temp;
			}
		}
	}
}


template <typename DType>
__global__ void Local_Diff_Backward_kernel(
	DeviceTensor<DType, 4> gradoutput,
	DeviceTensor<DType, 4> input_F,
	DeviceTensor<DType, 4> gradinput_F,
	int kernel_len) {
	//  input_Q: B, C, H, W   input_F: B, kernel_size, H, W
	// gradoutput: B, C, H, W (padded)
	int b = blockIdx.x;
	int C = input_F.getSize(1);
	int H = input_F.getSize(2);
	int W = input_F.getSize(3);
	int kernel_size = kernel_len * kernel_len;
	int pad_len = (kernel_len - 1) / 2;
	int kernel_center_index = (kernel_size - 1) / 2;
	DType temp;
	// ignore the edge points
	// the edge points of input_F are padding area (0).
	// gradoutput must have been padded.    * important
	for (int x = threadIdx.x; x < (H - kernel_len + 1) * (W - kernel_len + 1); x += blockDim.x) {
		int h = x / (W - kernel_len + 1) + pad_len;
		int w = x % (W - kernel_len + 1) + pad_len;
		for (int c = 0; c < C; ++c) {
			temp = gradinput_F[b][c][h][w];
			for (int index = 0; index < kernel_size; ++index) {
				if (index != kernel_center_index) {
					int i = index / kernel_len - pad_len;
					int j = index % kernel_len - pad_len;
					// grad from current position
					temp += gradoutput[b][index][h][w] * (input_F[b][c][h][w] - input_F[b][c][h + i][w + j]);
					// grad from neighboardhood position
					temp += gradoutput[b][index][h - i][w - j] * (input_F[b][c][h][w] - input_F[b][c][h - i][w - j]);
				}
			}
			gradinput_F[b][c][h][w] = temp;
		}
	}
}


at::Tensor Local_Diff_Forward_CUDA(
	const at::Tensor input_F_,
	int kernel_len) {
	int B = input_F_.size(0);
	int C = input_F_.size(1);
	int H = input_F_.size(2);
	int W = input_F_.size(3);
	int kernel_size = kernel_len * kernel_len;
	auto output_ = at::zeros({ B, kernel_size, H - kernel_len + 1, W - kernel_len + 1 }).to(input_F_.device(), input_F_.dtype());
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	dim3 blocks(B);
	dim3 threads(getNumThreads((H - kernel_len + 1) * (W - kernel_len + 1)));
	AT_DISPATCH_FLOATING_TYPES(input_F_.scalar_type(), "Local_CRF_Forward_CUDA", ([&] {
		/* Device tensors */
		DeviceTensor<scalar_t, 4> output = devicetensor<scalar_t, 4>(output_);
		DeviceTensor<scalar_t, 4> input_F = devicetensor<scalar_t, 4>(input_F_);
		/* kernel function */
		Local_Diff_Forward_kernel<scalar_t> << <blocks, threads, 0, stream >> > (
			output, input_F, kernel_len);
	}));
	AT_ASSERT(cudaGetLastError() == cudaSuccess);
	return output_;
}

at::Tensor Local_Diff_Backward_CUDA(
	const at::Tensor gradoutput_,
	const at::Tensor input_F_,
	int kernel_len) {
	/* outputs*/
	int B = input_F_.size(0);
	int C = input_F_.size(1);
	int H = input_F_.size(2);
	int W = input_F_.size(3);
	auto gradinput_F_ = at::zeros_like(input_F_);
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	dim3 blocks(B);
	dim3 threads(getNumThreads((H - kernel_len + 1) * (W - kernel_len + 1)));
	AT_DISPATCH_FLOATING_TYPES(input_F_.scalar_type(), "Local_CRF_Backward_CUDA", ([&] {
		/* Device tensors */
		DeviceTensor<scalar_t, 4> gradoutput = devicetensor<scalar_t, 4>(gradoutput_);
		DeviceTensor<scalar_t, 4> input_F = devicetensor<scalar_t, 4>(input_F_);
		DeviceTensor<scalar_t, 4> gradinput_F = devicetensor<scalar_t, 4>(gradinput_F_);
		/* kernel function */
		Local_Diff_Backward_kernel<scalar_t>
			<< <blocks, threads, 0, stream >> > (
				gradoutput, input_F, gradinput_F, kernel_len);
	}));
	AT_ASSERT(cudaGetLastError() == cudaSuccess);
	return gradinput_F_;
}