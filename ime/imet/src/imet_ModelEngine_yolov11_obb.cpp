#include "imet_ModelEngine_yolov11_obb.hpp"

#include"cuda_device_runtime_api.h"

#include<fstream>
#include<memory>

namespace rw {
	namespace imet {
		ModelEngine_yolov11_obb::ModelEngine_yolov11_obb(std::string model_path, nvinfer1::ILogger& logger)
		{
			init(model_path, logger);
		}

		ModelEngine_yolov11_obb::~ModelEngine_yolov11_obb()
		{
			for (int i = 0; i < 2; i++)
				(cudaFree(gpu_buffers[i]));
			delete[] cpu_output_buffer;

			delete context;
			delete engine;
			delete runtime;
		}

		void ModelEngine_yolov11_obb::preprocess(cv::Mat& image)
		{
			auto infer_image = cv::dnn::blobFromImage(image, 1.f / 255.f, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true);//1、缩放cv::resize;2、系数变换；3、色域变换bgr->rgb；4、图像裁剪cv::crop;5、数据标准化(x-mean)/var
			(cudaMemcpy(gpu_buffers[0], infer_image.data, input_w * input_h * image.channels() * sizeof(float), cudaMemcpyHostToDevice));
		}

		void ModelEngine_yolov11_obb::infer()
		{
			this->context->setInputTensorAddress(engine->getIOTensorName(0), gpu_buffers[0]);
			this->context->setOutputTensorAddress(engine->getIOTensorName(1), gpu_buffers[1]);
			this->context->enqueueV3(NULL);
		}

		void ModelEngine_yolov11_obb::postprocess(std::vector<Detection>& output)
		{
			(cudaMemcpy(cpu_output_buffer, gpu_buffers[1], num_detections * detection_attribute_size * sizeof(float), cudaMemcpyDeviceToHost));
			std::vector<cv::Rect> boxes;
			std::vector<int> class_ids;
			std::vector<float> confidences;

			const cv::Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);

			for (int i = 0; i < det_output.cols; ++i) {
				const  cv::Mat classes_scores = det_output.col(i).rowRange(4, 4 + num_classes);
				cv::Point class_id_point;
				double score;
				minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

				if (score > conf_threshold) {
					const float cx = det_output.at<float>(0, i);
					const float cy = det_output.at<float>(1, i);
					const float ow = det_output.at<float>(2, i);
					const float oh = det_output.at<float>(3, i);
					cv::Rect box;
					box.x = static_cast<int>((cx - 0.5 * ow));
					box.y = static_cast<int>((cy - 0.5 * oh));
					box.width = static_cast<int>(ow);
					box.height = static_cast<int>(oh);

					boxes.push_back(box);
					class_ids.push_back(class_id_point.y);
					confidences.push_back(score);

				}
			}
			std::vector<int> nms_result;
			cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);
			for (int i = 0; i < nms_result.size(); i++)
			{
				Detection result;
				int idx = nms_result[i];
				result.class_id = class_ids[idx];
				result.conf = confidences[idx];
				result.bbox = boxes[idx];
				output.push_back(result);
			}
		}

		void ModelEngine_yolov11_obb::draw(cv::Mat& image, const std::vector<Detection>& output)
		{
			auto ccc = cv::Mat::ones(cv::Size(10, 10), CV_8UC3);
			std::cout << cv::sum(ccc)[0] << std::endl;
			auto bb = cv::sum(ccc);
			const float ratio_h = input_h / (float)image.rows;
			const float ratio_w = input_w / (float)image.cols;
			for (int i = 0; i < output.size(); i++)
			{
				auto detection = output[i];
				auto box = detection.bbox;
				auto class_id = detection.class_id;
				auto conf = detection.conf;
				box.x = box.x / ratio_w;
				box.y = box.y / ratio_h;
				box.width = box.width / ratio_w;
				box.height = box.height / ratio_h;
				rectangle(image, cv::Point(box.x, box.y), cv::Point(box.x + box.width, box.y + box.height), { 0, 114, 189 }, 3);
				// Detection box text
				std::string class_string = std::to_string(class_id) + ' ' + std::to_string(conf).substr(0, 4);
				cv::Size text_size = cv::getTextSize(class_string, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
				cv::Rect text_rect(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
				rectangle(image, text_rect, { 0, 114, 189 }, cv::FILLED);
				putText(image, class_string, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
			}
		}

		void ModelEngine_yolov11_obb::init(std::string engine_path, nvinfer1::ILogger& logger)
		{
			std::ifstream engineStream(engine_path,std::ios::binary);
			engineStream.seekg(0, std::ios::end);
			const size_t modelSize = engineStream.tellg();
			engineStream.seekg(0, std::ios::beg);
			std::unique_ptr<char[]> engineData(new char[modelSize]);
			engineStream.read(engineData.get(), modelSize);
			engineStream.close();

			runtime = nvinfer1::createInferRuntime(logger);
			engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
			context = engine->createExecutionContext();

			input_h = engine->getTensorShape(engine->getIOTensorName(0)).d[2];
			input_w = engine->getTensorShape(engine->getIOTensorName(0)).d[3];
			detection_attribute_size = engine->getTensorShape(engine->getIOTensorName(1)).d[1];
			num_detections = engine->getTensorShape(engine->getIOTensorName(1)).d[2];
			num_classes = detection_attribute_size - 4;

			cpu_output_buffer = new float[num_detections * detection_attribute_size];
			(cudaMalloc((void**)&gpu_buffers[0], 3 * input_w * input_h * sizeof(float)));

			(cudaMalloc((void**)&gpu_buffers[1], detection_attribute_size * num_detections * sizeof(float)));

			for (int i = 0;i<10;i++) {
				this->infer();
			}
			cudaDeviceSynchronize();
		}

	}
}