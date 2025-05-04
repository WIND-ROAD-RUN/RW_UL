#include"imet_ModelEngine_yolov11_obb.hpp"

#include"cuda_device_runtime_api.h"

#include<fstream>
#include<memory>
#include <iomanip> 
#include <sstream> 

namespace rw
{
	namespace imet
	{
		ModelEngine_Yolov11_Obb::Detection::operator DetectionRectangleInfo() const
		{
			DetectionRectangleInfo result;
			result.width = bbox.width;
			result.height = bbox.height;

			result.leftTop.first = bbox.x;
			result.leftTop.second = bbox.y;

			result.rightTop.first = bbox.x + bbox.width;
			result.rightTop.second = bbox.y;

			result.leftBottom.first = bbox.x;
			result.leftBottom.second = bbox.y + bbox.height;

			result.rightBottom.first = bbox.x + bbox.width;
			result.rightBottom.second = bbox.y + bbox.height;

			result.center_x = bbox.x + bbox.width / 2;
			result.center_y = bbox.y + bbox.height / 2;

			result.area = bbox.width * bbox.height;
			result.classId = class_id;
			result.score = conf;

			return result;
		}

		ModelEngine_Yolov11_Obb::ModelEngine_Yolov11_Obb(const std::string& modelPath,
		                                                                   nvinfer1::ILogger& logger)
		{
			init(modelPath, logger);
		}

		ModelEngine_Yolov11_Obb::~ModelEngine_Yolov11_Obb()
		{
			for (int i = 0; i < 2; i++)
				(cudaFree(gpu_buffers[i]));
			delete[] cpu_output_buffer;
			delete context;
			delete engine;
			delete runtime;
		}

		void ModelEngine_Yolov11_Obb::init(std::string engine_path, nvinfer1::ILogger& logger)
		{
			std::ifstream engineStream(engine_path, std::ios::binary);
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

			for (int i = 0;i < 10;i++) {
				this->infer();
			}
			cudaDeviceSynchronize();
		}

		void ModelEngine_Yolov11_Obb::infer()
		{
			this->context->setInputTensorAddress(engine->getIOTensorName(0), gpu_buffers[0]);
			this->context->setOutputTensorAddress(engine->getIOTensorName(1), gpu_buffers[1]);
			this->context->enqueueV3(NULL);
		}

		std::vector<DetectionRectangleInfo> ModelEngine_Yolov11_Obb::postProcess()
		{
			std::vector<Detection> output;
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
			auto size = output.size();
			if (size==0){
				return {};
			}

			std::vector<DetectionRectangleInfo> result;
			result.reserve(size);
			for (const auto & item:output)
			{
				result.emplace_back(item);
			}

			return result;

		}

		cv::Mat ModelEngine_Yolov11_Obb::draw(const cv::Mat& mat, const std::vector<DetectionRectangleInfo>& infoList)
		{
			cv::Mat result = mat.clone();
			ImagePainter::PainterConfig config;
			for (const auto& item : infoList)
			{
				std::ostringstream oss;
				oss << "classId:"<<item.classId << " score:" << std::fixed << std::setprecision(2) << item.score;
				config.text = oss.str();
				ImagePainter::drawShapesOnSourceImg(result, item, config);
			}
			return result;
		}

		void ModelEngine_Yolov11_Obb::preprocess(const cv::Mat& mat)
		{
			sourceWidth = mat.cols;
			sourceHeight = mat.rows;
			auto infer_image = 
				cv::dnn::blobFromImage(mat, 
					1.f / 255.f,
					cv::Size(input_w, input_h),
					cv::Scalar(0, 0, 0), true);//1、缩放cv::resize;2、系数变换；3、色域变换bgr->rgb；4、图像裁剪cv::crop;5、数据标准化(x-mean)/var

			(cudaMemcpy(gpu_buffers[0],
				infer_image.data, 
				input_w * input_h * mat.channels() * sizeof(float), cudaMemcpyHostToDevice));
		}
	}
}
