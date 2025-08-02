#pragma once

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

namespace rw
{
	void launch_letterbox_kernel(
		const unsigned char* src, int src_w, int src_h, int src_stride,
		float* dst, int dst_w, int dst_h,
		float scale, int pad_w, int pad_h,
		unsigned char pad_b, unsigned char pad_g, unsigned char pad_r);

	void launch_letterbox_kernel(
		const unsigned char* src, int src_w, int src_h, int src_stride,
		float* dst, int dst_w, int dst_h,
		float scale, int pad_w, int pad_h,
		unsigned char pad_b, unsigned char pad_g, unsigned char pad_r,
		cudaStream_t stream);

	struct LetterBoxInfo
	{
		float letterBoxScale{};
		int pad_w{};
		int pad_h{};
	};

	struct LetterBoxConfig
	{
		float* dstDevData;
		int dstHeight;
		int dstWidth;
		unsigned char pad_b;
		unsigned char pad_g;
		unsigned char pad_r;
	};

	struct ImgPreprocess
	{
	public:
		static LetterBoxInfo LetterBox(const cv::Mat& srcImg, LetterBoxConfig& cfg, cudaStream_t stream);
		static LetterBoxInfo LetterBox(const cv::Mat& srcImg, LetterBoxConfig& cfg);
	};

}


