#pragma once

#include"ime_ModelEngine.h"

#include"NvInfer.h"

#include<string>

namespace rw {
	namespace imet {
		class ModelEngine_Yolov11_seg
			:public ModelEngine 
		{
		private:
			struct DetectionSeg
			{
				float conf{};
				int class_id{};
				cv::Rect bbox{};
			};
		private:
			void preprocess(const cv::Mat& mat) override;
			void infer() override;
			std::vector<DetectionRectangleInfo> postProcess() override;
		private:
			void init(std::string engine_path, nvinfer1::ILogger& logger);
		public:
			ModelEngine_Yolov11_seg(const std::string& modelPath, nvinfer1::ILogger& logger);
			~ModelEngine_Yolov11_seg() override;
		private:
			std::vector<DetectionRectangleInfo> convertDetectionToDetectionRectangleInfo(const std::vector<DetectionSeg>& detections);
		public:
			cv::Mat draw(const cv::Mat& mat, const std::vector<DetectionRectangleInfo>& infoList) override;

			float* gpu_buffers[3];               //!< The vector of device buffers needed for engine execution
			float* cpu_output_buffer;
			float* cpu_output_buffer2;
			nvinfer1::IRuntime* runtime;                 //!< The TensorRT runtime used to deserialize the engine
			nvinfer1::ICudaEngine* engine;               //!< The TensorRT engine used to run the network
			nvinfer1::IExecutionContext* context;        //!< The context for executing inference using an ICudaEngine

			// Model parameters
			int input_w;
			int input_h;
			int num_detections;
			int maskCoefficientNum;
			int detection_attribute_size;
			int num_classes = 80;
			float conf_threshold = 0.3f;
			float nms_threshold = 0.4f;
		private:
			std::vector<size_t> need_keep_classids{};
		public:
			void setConf_threshold(float num) {
				conf_threshold = num;
			}

			void setNms_threshold(float num) {
				nms_threshold = num;
			}
			void setNeed_keep_classids(std::vector<size_t> num) {
				need_keep_classids = num;
			}
		private:
			int sourceWidth{};
			int sourceHeight{};
		};
	
	
	
	}
}

