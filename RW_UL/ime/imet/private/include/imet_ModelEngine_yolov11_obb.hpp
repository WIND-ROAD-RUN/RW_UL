#pragma once

#include"ime_ModelEngine.h"
#include"ime_ModelEngineConfig.h"

#include"NvInfer.h"

#include<string>

namespace rw
{
	namespace imet
	{
		class ModelEngine_Yolov11_obb
			: public ModelEngine
		{
		public:
			struct Detection
			{
			public:
				double conf;
				int class_id;
				float angle;
				float x;       // x-coordinate of the center
				float y;       // y-coordinate of the center
				float width;   // width of the box
				float height;  // height of the box
			};
		public:
			ModelEngine_Yolov11_obb(const std::string& modelPath, nvinfer1::ILogger& logger);
		public:
			~ModelEngine_Yolov11_obb() override;
		private:
			void init(std::string engine_path, nvinfer1::ILogger& logger);
		private:
			void infer() override;
			std::vector<DetectionRectangleInfo> postProcess() override;
			void preprocess(const cv::Mat& mat) override;
		public:
			cv::Mat draw(const cv::Mat& mat, const std::vector<DetectionRectangleInfo>& infoList) override;
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

			ModelEngineConfig config;
			std::vector<DetectionRectangleInfo> convertDetectionToDetectionRectangleInfo(const std::vector<Detection>& detections);

		public:
			void setConfig(const ModelEngineConfig& config)
			{
				this->config = config;
			}
		private:
			int sourceWidth{};
			int sourceHeight{};
		};
	}
}
