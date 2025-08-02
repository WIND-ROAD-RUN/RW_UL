#include"imet_Utility.cuh"

namespace rw
{
	namespace imet
	{
		// CUDA kernel for transpose [attr, num] -> [num, attr]
		__global__ void transpose_kernel(const float* src, float* dst, int num, int attr, int total)
		{
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= total) return;
			// src: [attr, num], dst: [num, attr]
			int i = idx % num;    // column index in src
			int j = idx / num;    // row index in src
			dst[i * attr + j] = src[j * num + i];
		}

		void Utility::transpose(const float* src, float* dst, int num, int attr, cudaStream_t stream)
		{
			int total = num * attr;
			int block = 256;
			int grid = (total + block - 1) / block;
			transpose_kernel << <grid, block, 0, stream >> > (src, dst, num, attr, total);

		}

		__global__ void decode_kernel(
			const float* src, // [num, 4+num_classes]
			float* dst,       // [max_output, box_element]
			int num, int num_classes, float conf_thresh, int max_output, int box_element)
		{
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= num) return;

			const float* cur = src + idx * (4 + num_classes);

			float max_score = -1e10f;
			int max_class = -1;
			for (int c = 0; c < num_classes; ++c) {
				float score = cur[4 + c];
				if (score > max_score) {
					max_score = score;
					max_class = c;
				}
			}

			if (max_score < conf_thresh) return;

			if (idx < max_output) {
				float* out = dst + idx * box_element;
				out[0] = cur[0]; // x1
				out[1] = cur[1]; // y1
				out[2] = cur[2]; // x2
				out[3] = cur[3]; // y2
				out[4] = max_score; // conf
				out[5] = static_cast<float>(max_class); 
				out[6] = 1.0f; // keepflag, 1=keep, 0=ignore
			}
		}

		void Utility::decode(const float* src, float* dst, int num, int num_classes, float conf_thresh, int max_output,
			int box_element, cudaStream_t stream)
		{
			int block = 256;
			int grid = (num + block - 1) / block;
			decode_kernel << <grid, block, 0, stream >> > (src, dst, num, num_classes, conf_thresh, max_output, box_element);
		}

		__device__ float iou(const float* a, const float* b) {
			float x1 = fmaxf(a[0], b[0]);
			float y1 = fmaxf(a[1], b[1]);
			float x2 = fminf(a[2], b[2]);
			float y2 = fminf(a[3], b[3]);
			float w = fmaxf(0.0f, x2 - x1);
			float h = fmaxf(0.0f, y2 - y1);
			float inter = w * h;
			float area_a = fmaxf(0.0f, a[2] - a[0]) * fmaxf(0.0f, a[3] - a[1]);
			float area_b = fmaxf(0.0f, b[2] - b[0]) * fmaxf(0.0f, b[3] - b[1]);
			return inter / (area_a + area_b - inter + 1e-6f);
		}

		__global__ void nms_kernel(const float* boxes, float nms_thresh, int max_output, int box_element, int* keep_flag)
		{
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= max_output) return;

			const float* cur = boxes + idx * box_element;
			if (cur[4] == 0) { // 置信度为0的box直接跳过
				keep_flag[idx] = 0;
				return;
			}
			keep_flag[idx] = 1; // 先假设保留

			for (int i = 0; i < max_output; ++i) {
				if (i == idx) continue;
				const float* cmp = boxes + i * box_element;
				if (cmp[4] == 0) continue;
				// 只抑制同类别
				if (cur[5] == cmp[5] && cmp[4] > cur[4]) {
					if (iou(cur, cmp) > nms_thresh) {
						keep_flag[idx] = 0;
						return;
					}
				}
			}
		}

		__global__ void count_keep_kernel(const int* keep_flag, int max_output, int* keep_num)
		{
			int count = 0;
			for (int i = 0; i < max_output; ++i) {
				if (keep_flag[i]) ++count;
			}
			*keep_num = count;
		}


		void Utility::nms(const float* src, float nms_thresh, int max_output, int box_element, int* keep_flag,
			int* keep_num, cudaStream_t stream)
		{
			int block = 256;
			int grid = (max_output + block - 1) / block;
			nms_kernel << <grid, block, 0, stream >> > (src, nms_thresh, max_output, box_element, keep_flag);
			// 统计保留数量（可用thrust::reduce优化，这里用简单kernel）
			count_keep_kernel << <1, 1, 0, stream >> > (keep_flag, max_output, keep_num);
		}
	}
}
