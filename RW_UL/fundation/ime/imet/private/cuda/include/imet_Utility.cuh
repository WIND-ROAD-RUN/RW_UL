#pragma once

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

namespace rw
{
	namespace imet
	{
		struct Utility
		{
			void transpose(const float* src, float* dst, int num, int attr, cudaStream_t stream);
			void decode(const float* src, float* dst, int num, int num_classes, float conf_thresh, int max_output, int box_element, cudaStream_t stream);
			void nms(const float* src, float nms_thresh, int max_output, int box_element, int* keep_flag, int* keep_num, cudaStream_t stream);
		};
	}
}
