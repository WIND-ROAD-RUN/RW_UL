#pragma once

#include"imet_Utility.cuh"

namespace rw
{
	namespace imet
	{
		struct PostProcess
		{
			static void decode(float* src, float* dst, int numBboxes, int numClasses, float confThresh, int maxObjects, int numBoxElement, cudaStream_t stream);
		};
	}
}
