#include"imet_PostProcess.cuh"

#include "ime_ModelEngine.h"

namespace rw
{
	namespace imet
	{

		__global__ void decode_kernel(float* src, float* dst, int numBboxes, int numClasses, float confThresh, int maxObjects, int numBoxElement) {
			int position = blockDim.x * blockIdx.x + threadIdx.x;
			if (position >= numBboxes) return;

			float* pitem = src + (4 + numClasses) * position;
			float* classConf = pitem + 4;
			float confidence = 0;
			int label = 0;
			for (int i = 0; i < numClasses; i++) {
				if (classConf[i] > confidence) {
					confidence = classConf[i];
					label = i;
				}
			}

			if (confidence < confThresh) return;

			int index = (int)atomicAdd(dst, 1);
			if (index >= maxObjects) return;

			float cx = pitem[0];
			float cy = pitem[1];
			float width = pitem[2];
			float height = pitem[3];

			float left = cx - width * 0.5f;
			float top = cy - height * 0.5f;
			float right = cx + width * 0.5f;
			float bottom = cy + height * 0.5f;

			float* pout_item = dst + 1 + index * numBoxElement;
			pout_item[0] = left;
			pout_item[1] = top;
			pout_item[2] = right;
			pout_item[3] = bottom;
			pout_item[4] = confidence;
			pout_item[5] = label;
			pout_item[6] = 1;  // 1 = keep, 0 = ignore
		}

		void PostProcess::decode_det(float* src, float* dst, int numBboxes, int numClasses, float confThresh, int maxObjects, int numBoxElement, cudaStream_t stream)
		{
			cudaMemsetAsync(dst, 0, sizeof(int), stream);
			int blockSize = 256;
			int gridSize = (numBboxes + blockSize - 1) / blockSize;
			decode_kernel << <gridSize, blockSize, 0, stream >> > (src, dst, numBboxes, numClasses, confThresh, maxObjects, numBoxElement);
		}

		__global__ void remove_mask_coeff_kernel(
			const float* src, float* dst, int numBboxes, int detElementNum, int srcElementNum)
		{
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx < numBboxes) {
				const float* srcPtr = src + idx * srcElementNum;
				float* dstPtr = dst + idx * detElementNum;
				// 只拷贝det相关部分
				for (int i = 0; i < detElementNum; ++i) {
					dstPtr[i] = srcPtr[i];
				}
			}
		}

		void PostProcess::decode_seg(float* src, float* dst, int numBboxes, int numClasses, float confThresh,
			int maxObjects, int numBoxElement, cudaStream_t stream)
		{
			// 计算每个bbox的有效元素数量（去除mask系数）
			int detElementNum = 4  + numClasses; // 4 bbox + 1 conf + numClasses
			int srcElementNum = detElementNum + MaskCoefficientNum;

			// 分配临时buffer用于存放去除mask后的数据
			float* detBuffer = nullptr;
			cudaMalloc(&detBuffer, numBboxes * detElementNum * sizeof(float));

			// 调用kernel去除mask系数
			int blockSize = 256;
			int gridSize = (numBboxes + blockSize - 1) / blockSize;
			remove_mask_coeff_kernel << <gridSize, blockSize, 0, stream >> > (
				src, detBuffer, numBboxes, detElementNum, srcElementNum);

			// 复用det的decode核心逻辑
			decode_kernel << <gridSize, blockSize, 0, stream >> > (
				detBuffer, dst, numBboxes, numClasses, confThresh, maxObjects, numBoxElement);

			cudaFree(detBuffer);
		}
	}
}
