#include"imet_ModelEngine_yolov11_obb.hpp"

#include"cuda_device_runtime_api.h"

#include<fstream>
#include<memory>
#include <iomanip> 
#include <sstream>
#include <algorithm>
#include <numeric>

namespace rw
{
	namespace imet
	{

		ModelEngine_Yolov11_obb::ModelEngine_Yolov11_obb(const std::string& modelPath,
			nvinfer1::ILogger& logger)
		{
			init(modelPath, logger);
		}

		ModelEngine_Yolov11_obb::~ModelEngine_Yolov11_obb()
		{
			for (int i = 0; i < 2; i++)
				(cudaFree(gpu_buffers[i]));
			delete[] cpu_output_buffer;
			delete context;
			delete engine;
			delete runtime;
		}

		void ModelEngine_Yolov11_obb::init(std::string engine_path, nvinfer1::ILogger& logger)
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
			num_classes = detection_attribute_size - 5;

			cpu_output_buffer = new float[num_detections * detection_attribute_size];
			(cudaMalloc((void**)&gpu_buffers[0], 3 * input_w * input_h * sizeof(float)));

			(cudaMalloc((void**)&gpu_buffers[1], detection_attribute_size * num_detections * sizeof(float)));

			for (int i = 0;i < 10;i++) {
				this->infer();
			}
			cudaDeviceSynchronize();
		}

		void ModelEngine_Yolov11_obb::infer()
		{
			this->context->setInputTensorAddress(engine->getIOTensorName(0), gpu_buffers[0]);
			this->context->setOutputTensorAddress(engine->getIOTensorName(1), gpu_buffers[1]);
			this->context->enqueueV3(NULL);
		}


		std::vector<DetectionRectangleInfo> ModelEngine_Yolov11_obb::postProcess()
		{
			(cudaMemcpy(cpu_output_buffer, gpu_buffers[1], num_detections * detection_attribute_size * sizeof(float), cudaMemcpyDeviceToHost));
			std::vector<Detection> boxes;

			const cv::Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);

			for (int i = 0; i < det_output.cols; ++i) {
				const  cv::Mat classes_scores = det_output.col(i).rowRange(4, 4 + num_classes);
				cv::Point class_id_point;
				double score;
				minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

				if (score > config.conf_threshold) {
					const float cx = det_output.at<float>(0, i);
					const float cy = det_output.at<float>(1, i);
					const float ow = det_output.at<float>(2, i);
					const float oh = det_output.at<float>(3, i);
					const float angle= det_output.at<float>(4 + num_classes, i);
					Detection box;
                    box.angle = angle;
                    box.c_x = cx;
                    box.c_y = cy;
                    box.width = ow;
                    box.height = oh;
                    box.conf = score;
                    box.class_id = class_id_point.y;
					boxes.push_back(box);
				}
			}
			std::vector<Detection> nms_boxes = rotatedNMS(boxes, config.nms_threshold);

			auto result = convertDetectionToDetectionRectangleInfo(nms_boxes);

			return result;
		}

		cv::Mat ModelEngine_Yolov11_obb::draw(const cv::Mat& mat, const std::vector<DetectionRectangleInfo>& infoList)
		{
			cv::Mat result = mat.clone();
			ImagePainter::PainterConfig config;
			for (const auto& item : infoList)
			{
				std::ostringstream oss;
				oss << "classId:" << item.classId << " score:" << std::fixed << std::setprecision(2) << item.score;
				config.text = oss.str();
				ImagePainter::drawShapesOnSourceImg(result, item, config);
			}
			return result;
		}

		std::vector<DetectionRectangleInfo> ModelEngine_Yolov11_obb::convertDetectionToDetectionRectangleInfo(
			const std::vector<Detection>& detections)
		{
			std::vector<DetectionRectangleInfo> result;
			result.reserve(detections.size());
			std::vector<Detection> postDections;
			if (config.imagePretreatmentPolicy == ImagePretreatmentPolicy::Resize)
			{
				postDections = convertWhenResize(detections);
			}
			else if (config.imagePretreatmentPolicy == ImagePretreatmentPolicy::LetterBox)
			{
				postDections = convertWhenLetterBox(detections);
			}
			else if (config.imagePretreatmentPolicy == ImagePretreatmentPolicy::CenterCrop)
			{
				postDections = convertWhenCentralCrop(detections);
			}
			else
			{
				postDections = convertWhenResize(detections);
			}


			for (const auto& item : postDections)
			{
				DetectionRectangleInfo resultItem;
			
				float cx = item.c_x;
				float cy = item.c_y;
				float w = item.width;
				float h = item.height;
				float angle_rad = item.angle;

				float dx[4] = { -w / 2,  w / 2,  w / 2, -w / 2 };
				float dy[4] = { -h / 2, -h / 2,  h / 2,  h / 2 };

				std::pair<int, int> corners[4];
				for (int i = 0; i < 4; ++i) {
					float x_rot = dx[i] * std::cos(angle_rad) - dy[i] * std::sin(angle_rad);
					float y_rot = dx[i] * std::sin(angle_rad) + dy[i] * std::cos(angle_rad);
					corners[i].first = static_cast<int>(std::round(cx + x_rot));
					corners[i].second = static_cast<int>(std::round(cy + y_rot));
				}

				resultItem.leftTop = corners[0];
				resultItem.rightTop = corners[1];
				resultItem.rightBottom = corners[2];
				resultItem.leftBottom = corners[3];
				resultItem.center_x = static_cast<int>(std::round(cx));
				resultItem.center_y = static_cast<int>(std::round(cy));
				resultItem.width = static_cast<int>(std::round(w));
				resultItem.height = static_cast<int>(std::round(h));
				resultItem.area = static_cast<long>(std::round(w * h));
				resultItem.classId = static_cast<size_t>(item.class_id);
				resultItem.score = item.conf;

				result.push_back(resultItem);
			}
			return result;
		}

		std::vector<ModelEngine_Yolov11_obb::Detection> ModelEngine_Yolov11_obb::convertWhenResize(
			const std::vector<Detection>& detections)
		{
			std::vector<Detection> result;
			result.reserve(detections.size());
			auto scaleX = sourceWidth / static_cast<float>(input_w);
			auto scaleY = sourceHeight / static_cast<float>(input_h);
			for (const auto& item : detections)
			{
				Detection resultItem;
				resultItem.width = item.width * scaleX;
				resultItem.height = item.height * scaleY;
				resultItem.c_x = item.c_x * scaleX;
				resultItem.c_y = item.c_y * scaleY;
				resultItem.angle = item.angle;
				resultItem.conf = item.conf;
				resultItem.class_id = item.class_id;
				result.push_back(resultItem);
			}
			return result;
		}

		std::vector<ModelEngine_Yolov11_obb::Detection> ModelEngine_Yolov11_obb::convertWhenLetterBox(
			const std::vector<Detection>& detections)
		{
			std::vector<Detection> result;
			result.reserve(detections.size());
			const auto& params = centerCropParams;
			float scale = letterBoxScale;
			int dw = letterBoxdw;
			int dh = letterBoxdh;
			for (const auto& item : detections)
			{
				float x1 = (item.c_x - dw) / scale;
				float y1 = (item.c_y - dh) / scale;
				float x2 = (item.c_x + item.width - dw) / scale;
				float y2 = (item.c_y + item.height - dh) / scale;
				Detection resultItem;
				resultItem.c_x = x1;
				resultItem.c_y = y1;
				resultItem.width = x2 - x1;
				resultItem.height = y2 - y1;
				resultItem.angle = item.angle;
				resultItem.conf = item.conf;
				resultItem.class_id = item.class_id;
				result.push_back(resultItem);
			}
			return result;
		}

		std::vector<ModelEngine_Yolov11_obb::Detection> ModelEngine_Yolov11_obb::convertWhenCentralCrop(
			const std::vector<Detection>& detections)
		{
			std::vector<Detection> result;
			result.reserve(detections.size());
			const auto& params = centerCropParams;
			for (const auto& item : detections)
			{
				float x1 = item.c_x + params.crop_x - params.pad_left;
				float y1 = item.c_y + params.crop_y - params.pad_top;
				float x2 = item.c_x + item.width + params.crop_x - params.pad_left;
				float y2 = item.c_y + item.height + params.crop_y - params.pad_top;

				Detection resultItem;
				resultItem.c_x = (x1 + x2) / 2;
				resultItem.c_y = (y1 + y2) / 2;
				resultItem.width = x2 - x1;
				resultItem.height = y2 - y1;
				resultItem.angle = item.angle;
				resultItem.conf = item.conf;
				resultItem.class_id = item.class_id;
				result.push_back(resultItem);
			}
			return result;
		}

		void ModelEngine_Yolov11_obb::preprocess(const cv::Mat& mat)
		{
			sourceWidth = mat.cols;
			sourceHeight = mat.rows;
			if (config.imagePretreatmentPolicy == ImagePretreatmentPolicy::Resize)
			{
				auto infer_image = cv::dnn::blobFromImage(mat, 1.f / 255.f, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true);
				(cudaMemcpy(gpu_buffers[0], infer_image.data, input_w * input_h * mat.channels() * sizeof(float), cudaMemcpyHostToDevice));
			}
			else if (config.imagePretreatmentPolicy == ImagePretreatmentPolicy::LetterBox)
			{
				cv::Mat letterbox_image = PreProcess::letterbox(mat, input_w, input_h, config.letterBoxColor, letterBoxScale, letterBoxdw, letterBoxdh);
				auto infer_image = cv::dnn::blobFromImage(letterbox_image, 1.f / 255.f, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true);
				(cudaMemcpy(gpu_buffers[0], infer_image.data, input_w * input_h * mat.channels() * sizeof(float), cudaMemcpyHostToDevice));
			}
			else if (config.imagePretreatmentPolicy == ImagePretreatmentPolicy::CenterCrop)
			{
				cv::Mat center_crop_image = PreProcess::centerCrop(mat, input_w, input_h, config.centerCropColor, &centerCropParams);
				auto infer_image = cv::dnn::blobFromImage(center_crop_image, 1.f / 255.f, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true);
				(cudaMemcpy(gpu_buffers[0], infer_image.data, input_w * input_h * mat.channels() * sizeof(float), cudaMemcpyHostToDevice));
			}
			else
			{
				auto infer_image = cv::dnn::blobFromImage(mat, 1.f / 255.f, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true);
				(cudaMemcpy(gpu_buffers[0], infer_image.data, input_w * input_h * mat.channels() * sizeof(float), cudaMemcpyHostToDevice));
			}
		}

		std::vector<ModelEngine_Yolov11_obb::Detection> ModelEngine_Yolov11_obb::rotatedNMS(
			const std::vector<ModelEngine_Yolov11_obb::Detection>& dets, double iouThreshold)
		{
			std::vector<ModelEngine_Yolov11_obb::Detection> result;
			if (dets.empty()) return result;

			// ∞¥÷√–≈∂»Ωµ–Ú≈≈–Ú
			std::vector<size_t> idxs(dets.size());
			std::iota(idxs.begin(), idxs.end(), 0);
			std::sort(idxs.begin(), idxs.end(), [&](size_t i, size_t j) {
				return dets[i].conf > dets[j].conf;
				});

			std::vector<bool> suppressed(dets.size(), false);

			for (size_t i = 0; i < idxs.size(); ++i) {
				if (suppressed[idxs[i]]) continue;
				result.push_back(dets[idxs[i]]);
				for (size_t j = i + 1; j < idxs.size(); ++j) {
					if (suppressed[idxs[j]]) continue;
					if (rotatedIoU(dets[idxs[i]], dets[idxs[j]]) > iouThreshold) {
						suppressed[idxs[j]] = true;
					}
				}
			}
			return result;
		}

		double ModelEngine_Yolov11_obb::rotatedIoU(const ModelEngine_Yolov11_obb::Detection& a,
			const ModelEngine_Yolov11_obb::Detection& b)
		{
			cv::RotatedRect rect1 = toRotatedRect(a);
			cv::RotatedRect rect2 = toRotatedRect(b);

			std::vector<cv::Point2f> intersection;
			auto interType = cv::rotatedRectangleIntersection(rect1, rect2, intersection);

			if (interType == cv::INTERSECT_NONE || intersection.empty())
				return 0.0;

			double interArea = cv::contourArea(intersection);
			double area1 = rect1.size.area();
			double area2 = rect2.size.area();
			double iou = interArea / (area1 + area2 - interArea);
			return iou;
		}

		cv::RotatedRect ModelEngine_Yolov11_obb::toRotatedRect(const ModelEngine_Yolov11_obb::Detection& det)
		{
			return cv::RotatedRect(
				cv::Point2f(det.c_x + det.width / 2.0f, det.c_y + det.height / 2.0f), // center
				cv::Size2f(det.width, det.height),
				det.angle
			);
		}
	}
}
