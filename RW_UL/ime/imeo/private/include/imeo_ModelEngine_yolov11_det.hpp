#pragma once
#include"opencv2/opencv.hpp"

#include <string>
#include <locale>
#include <codecvt>

#include"ime_ModelEngine.h"
#include"onnxruntime_cxx_api.h"

#include"ime_ModelEngineConfig.h"

namespace rw {
    namespace imeo {
        class ModelEngine_Yolov11_det
	        :public ModelEngine{
        private:
            struct Detection
            {
                float conf;
                int class_id;
                cv::Rect bbox;
            };
        public:
            ModelEngine_Yolov11_det(const std::string& modelPath);
            ~ModelEngine_Yolov11_det() override;
        public:
            std::string  input_name;
            std::string  output_name;
        private:
            void preprocess(const cv::Mat& mat) override;
            void infer() override;
            std::vector<DetectionRectangleInfo> postProcess() override;
        private:
            void init(const std::string & engine_path);
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
            std::vector<DetectionRectangleInfo> convertDetectionToDetectionRectangleInfo(const std::vector<Detection>& detections);

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
        };

    }
}