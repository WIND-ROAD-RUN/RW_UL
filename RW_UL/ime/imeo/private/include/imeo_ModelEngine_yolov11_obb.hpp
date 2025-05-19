#pragma once
#include"opencv2/opencv.hpp"

#include <string>
#include <locale>

#include"ime_ModelEngine.h"
#include"ime_ModelEngineConfig.h"
#include"ime_utilty_private.hpp"

#include"onnxruntime_cxx_api.h"

namespace rw {
	namespace imeo {
		class ModelEngine_Yolov11_obb
			:public ModelEngine {
		private:
			struct Detection
			{
			public:
				double conf;
				int class_id;
				float angle;
				float c_x;       // x-coordinate of the center
				float c_y;       // y-coordinate of the center
				float width;   // width of the box
				float height;  // height of the box
			};
		public:
			ModelEngine_Yolov11_obb(const std::string& modelPath);
			~ModelEngine_Yolov11_obb() override;
		public:
			std::string  input_name;
			std::string  output_name;
		private:
			void preprocess(const cv::Mat& mat) override;
			void infer() override;
			std::vector<DetectionRectangleInfo> postProcess() override;
		private:
			void init(const std::string& engine_path);
		private:
			Ort::Env env;
			Ort::Session session = Ort::Session(nullptr);
			std::vector<Ort::Value>output_tensors;
			std::vector<const char*> input_node_names;
			std::vector<const char*> output_node_names;
			Ort::Value input_tensor = Ort::Value(nullptr);
			std::vector<Ort::Value> ort_inputs;
			cv::Mat infer_image;
		private:
			float* cpu_output_buffer;
			int input_w;
			int input_h;
			int num_detections;
			int detection_attribute_size;
			int num_classes = 80;

			ModelEngineConfig config;
		private:
			std::vector<Detection> rotatedNmsWithKeepClass(
				const std::vector<Detection>& dets,
				float conf_threshold,
				float nms_threshold,
				const std::vector<size_t>& need_keep_classids);
		private:
			std::vector<DetectionRectangleInfo> convertDetectionToDetectionRectangleInfo(const std::vector<Detection>& detections);
			std::vector<Detection> convertWhenResize(const std::vector<Detection>& detections);
			std::vector<Detection> convertWhenLetterBox(const std::vector<Detection>& detections);
			std::vector<Detection> convertWhenCentralCrop(const std::vector<Detection>& detections);
		private:
			std::vector<ModelEngine_Yolov11_obb::Detection> rotatedNMS(const std::vector<ModelEngine_Yolov11_obb::Detection>& dets, double iouThreshold);
			double rotatedIoU(const ModelEngine_Yolov11_obb::Detection& a, const ModelEngine_Yolov11_obb::Detection& b);
			cv::RotatedRect toRotatedRect(const ModelEngine_Yolov11_obb::Detection& det);
		public:
			cv::Mat draw(const cv::Mat& mat, const std::vector<DetectionRectangleInfo>& infoList) override;

		private:
			int sourceWidth{};
			int sourceHeight{};
		private:
			static std::wstring stringToWString(const std::string& str);
		public:
			void setConfig(const ModelEngineConfig& modelConfig)
			{
				this->config = modelConfig;
			}
		private:
			float letterBoxScale{};
			int letterBoxdw{};
			int letterBoxdh{};
		private:
			PreProcess::CenterCropParams centerCropParams;
		};
	}
}