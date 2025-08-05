#include"imet_PostProcess.cuh"

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

		void PostProcess::decode_seg(float* src, float* dst, int numBboxes, int numClasses, float confThresh,
			int maxObjects, int numBoxElement, cudaStream_t stream)
		{
			cudaMemsetAsync(dst, 0, sizeof(int), stream);
			int blockSize = 256;
			int gridSize = (numBboxes + blockSize - 1) / blockSize;
			decode_kernel << <gridSize, blockSize, 0, stream >> > (src, dst, numBboxes, numClasses, confThresh, maxObjects, numBoxElement);
		}
	}
}
