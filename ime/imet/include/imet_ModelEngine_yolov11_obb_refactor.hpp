#pragma once

#include"ime_ModelEngine.h"

#include"NvInfer.h"

#include<string>

namespace rw
{
	namespace imet
	{
		class ModelEngine_Yolov11_Obb_Refactor
			: public ModelEngine
		{
		private:
			struct Detection
			{
				float conf;
				int class_id;
				cv::Rect bbox;
			};
		public:
			ModelEngine_Yolov11_Obb_Refactor(const std::string & modelPath, nvinfer1::ILogger& logger);
		public:
			~ModelEngine_Yolov11_Obb_Refactor() override;
		private:
			void init(std::string engine_path, nvinfer1::ILogger& logger);
		private:
			void preprocess() override;
			void infer() override;
			DetectionRectangleInfo postProcess() override;
			cv::Mat draw(const cv::Mat& mat) override;
		private:
			float* gpu_buffers[2];               //!< The vector of device buffers needed for engine execution
			float* cpu_output_buffer;
			nvinfer1::IRuntime* runtime;                 //!< The TensorRT runtime used to deserialize the engine
			nvinfer1::ICudaEngine* engine;               //!< The TensorRT engine used to run the network
			nvinfer1::IExecutionContext* context;        //!< The context for executing inference using an ICudaEngine

			// Model parameters
			int input_w;
			int input_h;
			int num_detections;
			int detection_attribute_size;
			int num_classes = 80;
			float conf_threshold = 0.3f;
			float nms_threshold = 0.4f;

			std::vector<cv::Scalar> colors;
		};
	}
}
