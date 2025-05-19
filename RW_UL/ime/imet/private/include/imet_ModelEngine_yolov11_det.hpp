#pragma once

#include"ime_ModelEngine.h"
#include"ime_ModelEngineConfig.h"

#include"ime_utilty_private.hpp"

#include"NvInfer.h"

#include<string>

namespace rw
{
	namespace imet
	{
		class ModelEngine_Yolov11_det
			: public ModelEngine
		{
		private:
			struct Detection
			{
			public:
				float conf;
				int class_id;
				cv::Rect bbox;
			};
		public:
			ModelEngine_Yolov11_det(const std::string& modelPath, nvinfer1::ILogger& logger);
		public:
			~ModelEngine_Yolov11_det() override;
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
		public:
			void setConfig(const ModelEngineConfig& config)
			{
				this->config = config;
			}
		private:
			std::vector<DetectionRectangleInfo> convertDetectionToDetectionRectangleInfo(const std::vector<Detection>& detections);
			std::vector<DetectionRectangleInfo> convertWhenResize(const std::vector<Detection>& detections);
			std::vector<DetectionRectangleInfo> convertWhenLetterBox(const std::vector<Detection>& detections);
			std::vector<DetectionRectangleInfo> convertWhenCentralCrop(const std::vector<Detection>& detections);
		private:
			int sourceWidth{};
			int sourceHeight{};
		private:
			float letterBoxScale{};
			int letterBoxdw{};
			int letterBoxdh{};
		private:
			PreProcess::CenterCropParams centerCropParams;
		};
	}
}