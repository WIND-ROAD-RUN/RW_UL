#pragma once

#include"opencv2/opencv.hpp"

#include"NvInfer.h"
#include<string>

struct DetectionSeg
{
    float conf;
    int class_id;
    cv::Rect bbox;
};

namespace rw {
	namespace imet {


		class ModelEngine_yolov11_seg {
		public:
			ModelEngine_yolov11_seg(std::string model_path, nvinfer1::ILogger& logger);
			~ModelEngine_yolov11_seg();
        public:
            void init(std::string engine_path, nvinfer1::ILogger& logger);
            void preprocess(cv::Mat& image);
            void infer();
            void postprocess(std::vector<DetectionSeg>& output);
            void draw(cv::Mat& image, const std::vector<DetectionSeg>& output);
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