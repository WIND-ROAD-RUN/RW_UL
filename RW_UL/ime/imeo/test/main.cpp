#include"onnxruntime_cxx_api.h"

#include"opencv2/opencv.hpp"

#include<iostream>
#include<vector>
#include<string>

int main()
{
	Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "test");

	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(20);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	OrtCUDAProviderOptions cuda_options;
	cuda_options.device_id = 0; 
	cuda_options.gpu_mem_limit = SIZE_MAX;
	cuda_options.do_copy_in_default_stream = 1;

	session_options.AppendExecutionProvider_CUDA(cuda_options);

	std::wstring model_path = LR"(C:\Users\rw\Desktop\model\best.onnx)";
	Ort::Session session(env, model_path.c_str(), session_options);
}